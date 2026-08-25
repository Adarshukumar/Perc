# server.py
"""
Perchance Image Generation API v3.0
════════════════════════════════════

Lightweight, self-healing, scalable image generation server.
Handles thousands of concurrent users on Hugging Face Spaces.

Key fix: Server starts accepting connections IMMEDIATELY.
         Browser key fetch runs in BACKGROUND after startup.
"""

# ═══════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════

import asyncio
import base64
import json
import logging
import os
import random
import string
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import cloudscraper
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import zendriver as zd
from zendriver import cdp

try:
    from pyvirtualdisplay import Display as _VDisplay
    _VDISPLAY_AVAILABLE = True
except ImportError:
    _VDisplay = None
    _VDISPLAY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

API_BASE     = "https://image-generation.perchance.org"
API_GENERATE = "/api/generate"
API_DOWNLOAD = "/api/downloadTemporaryImage"
API_AWAIT    = "/api/awaitExistingGenerationRequest"
API_AD_CODE  = "/api/getAccessCodeForAdPoweredStuff"

BROWSER_TARGET   = "https://perchance.org/ai-text-to-image-generator"
BROWSER_ORIGIN   = "https://image-generation.perchance.org"
BROWSER_HEADLESS = os.environ.get("ZD_HEADLESS", "false").lower() in ("1", "true")
BROWSER_TIMEOUT  = 90
CLICK_INTERVAL   = 0.35
CLICK_JITTER     = 8.0

USE_VIRTUAL_DISPLAY = os.environ.get("USE_VIRTUAL_DISPLAY", "true").lower() in ("1", "true")

HTTP_TIMEOUT       = 30
DOWNLOAD_TIMEOUT   = 180
BACKOFF_BASE       = 0.7
GEN_MAX_RETRIES    = 6

KEY_MAX_RETRIES    = 3
KEY_COOLDOWN       = 30
KEY_MAX_FAILURES   = 5

WORKER_COUNT       = 3
QUEUE_CAPACITY     = 5000
THREAD_POOL_SIZE   = 16

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TASK_TTL           = 3600
CLEANUP_INTERVAL   = 300
MAX_STORED_TASKS   = 10_000

SSE_QUEUE_DEPTH    = 256
SSE_PING_SEC       = 15

PERCHANCE_HEADERS = {
    "Accept":       "*/*",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin":       BROWSER_ORIGIN,
    "Referer":      f"{BROWSER_ORIGIN}/embed",
    "User-Agent":   (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


# ═══════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("perchance")


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class GenerationRequest(BaseModel):
    prompts:         List[str] = Field(default_factory=list)
    prompt:          Optional[str] = Field(None)
    count:           int   = Field(1, ge=1, le=20)
    resolution:      str   = Field("512x768")
    guidance_scale:  float = Field(7.0, ge=1.0, le=20.0)
    negative_prompt: str   = Field("")
    style:           str   = Field("private")


class GenerationResponse(BaseModel):
    task_id:      str
    stream:       str
    status_url:   str
    total_images: int
    created_at:   str
    queue_pos:    int


class SetKeyRequest(BaseModel):
    userKey: str = Field(..., min_length=1)


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

_SAFE = set(string.ascii_letters + string.digits + "-_.()")


def _fname(s: str, limit: int = 80) -> str:
    return "".join(c if c in _SAFE else "_" for c in s)[:limit]


def _sid(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rid() -> str:
    return f"{time.time():.6f}-{_sid()}"


def _age(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return time.time() - dt.timestamp()
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════
#  VIRTUAL DISPLAY MANAGER
# ═══════════════════════════════════════════════════════════════

class DisplayManager:
    __slots__ = ("_disp",)

    def __init__(self):
        self._disp = None

    def ensure(self, headless: bool) -> None:
        if headless or not USE_VIRTUAL_DISPLAY:
            return
        if os.environ.get("DISPLAY"):
            return
        if not _VDISPLAY_AVAILABLE:
            log.warning("pyvirtualdisplay not installed — no DISPLAY available")
            return
        if self._disp is not None:
            return
        try:
            self._disp = _VDisplay(visible=0, size=(1280, 720))
            self._disp.start()
            log.info("Xvfb started  DISPLAY=%s", os.environ.get("DISPLAY"))
        except Exception as exc:
            self._disp = None
            log.exception("Xvfb start failed: %s", exc)

    def shutdown(self) -> None:
        if self._disp is None:
            return
        try:
            self._disp.stop()
            log.info("Xvfb stopped")
        except Exception:
            log.exception("Xvfb stop error")
        finally:
            self._disp = None


_display = DisplayManager()


# ═══════════════════════════════════════════════════════════════
#  SSE BROADCASTER
# ═══════════════════════════════════════════════════════════════

class Broadcaster:
    def __init__(self):
        self._subs: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_DEPTH)
        async with self._lock:
            self._subs.setdefault(task_id, []).append(q)
        return q

    async def unsubscribe(self, task_id: str, q: asyncio.Queue) -> None:
        async with self._lock:
            bucket = self._subs.get(task_id)
            if bucket:
                self._subs[task_id] = [x for x in bucket if x is not q]
                if not self._subs[task_id]:
                    del self._subs[task_id]

    async def emit(self, task_id: str, event: dict) -> None:
        async with self._lock:
            targets = list(self._subs.get(task_id, []))
        for q in targets:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def emit_global(self, event: dict) -> None:
        async with self._lock:
            all_targets = [
                (tid, list(qs)) for tid, qs in self._subs.items()
            ]
        for _, qs in all_targets:
            for q in qs:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    @property
    def total_subscribers(self) -> int:
        return sum(len(v) for v in self._subs.values())


# ═══════════════════════════════════════════════════════════════
#  TASK STORE
# ═══════════════════════════════════════════════════════════════

class TaskStore:
    def __init__(self):
        self._data: Dict[str, dict] = {}

    def create(self, *, prompts, count, resolution, guidance, negative, style) -> dict:
        tid = str(uuid.uuid4())
        task = {
            "id":            tid,
            "prompts":       prompts,
            "count":         count,
            "resolution":    resolution,
            "guidance":      guidance,
            "negative":      negative,
            "style":         style,
            "created_at":    _now(),
            "started_at":    None,
            "finished_at":   None,
            "status":        "queued",
            "total_images":  len(prompts) * count,
            "completed":     0,
            "failed_count":  0,
            "results":       [],
            "error":         None,
        }
        if len(self._data) >= MAX_STORED_TASKS:
            self._evict()
        self._data[tid] = task
        return task

    def get(self, tid: str) -> Optional[dict]:
        return self._data.get(tid)

    @property
    def size(self) -> int:
        return len(self._data)

    def active_count(self) -> int:
        return sum(1 for t in self._data.values() if t["status"] in ("queued", "running"))

    def _evict(self) -> None:
        finished = sorted(
            ((k, v) for k, v in self._data.items()
             if v["status"] in ("completed", "failed")),
            key=lambda x: x[1]["created_at"],
        )
        for tid, _ in finished[:max(1, len(finished) // 4)]:
            del self._data[tid]

    def purge_expired(self) -> int:
        stale = [
            tid for tid, t in self._data.items()
            if t["status"] in ("completed", "failed") and _age(t["created_at"]) > TASK_TTL
        ]
        for tid in stale:
            del self._data[tid]
        return len(stale)


# ═══════════════════════════════════════════════════════════════
#  KEY MANAGER
# ═══════════════════════════════════════════════════════════════

class KeyManager:
    def __init__(self):
        self.key: Optional[str]  = None
        self._rw_lock            = asyncio.Lock()
        self._usable             = asyncio.Event()
        self._usable.set()
        self._refresh_gate       = asyncio.Lock()
        self._last_ok: float     = 0.0
        self._consec_fails: int  = 0
        self._ready              = asyncio.Event()   # set once first key obtained

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def wait_ready(self, timeout: float = None):
        """Wait until at least one key has been obtained."""
        if timeout:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        else:
            await self._ready.wait()

    async def get(self) -> Optional[str]:
        await self._usable.wait()
        async with self._rw_lock:
            return self.key

    async def set(self, new_key: str) -> None:
        async with self._rw_lock:
            self.key = new_key
        self._last_ok      = time.time()
        self._consec_fails = 0
        self._usable.set()
        self._ready.set()
        log.info("userKey set OK (len=%d)", len(new_key))

    def reset_failures(self) -> None:
        self._consec_fails = 0

    async def refresh(self, broadcaster: Broadcaster) -> Optional[str]:
        async with self._refresh_gate:
            if self.key and (time.time() - self._last_ok) < KEY_COOLDOWN:
                log.info("Key refreshed %.0fs ago → reuse", time.time() - self._last_ok)
                return self.key

            if self._consec_fails >= KEY_MAX_FAILURES:
                msg = (
                    f"Auto-refresh disabled after {self._consec_fails} failures. "
                    "Set key manually via POST /v1/keys/set"
                )
                log.error(msg)
                await broadcaster.emit_global({
                    "event": "key:error", "message": msg, "time": _now(),
                })
                return None

            self._usable.clear()
            log.info("Browser key refresh starting …")
            await broadcaster.emit_global({
                "event": "key:refresh",
                "message": "UserKey expired — launching browser refresh …",
                "time": _now(),
            })

            try:
                new_key = await extract_key_via_browser(
                    timeout=BROWSER_TIMEOUT, headless=BROWSER_HEADLESS,
                )
                if new_key:
                    await self.set(new_key)
                    await broadcaster.emit_global({
                        "event": "key:ready",
                        "message": "UserKey refreshed — generation resuming",
                        "time": _now(),
                    })
                    return new_key

                self._consec_fails += 1
                log.error("Key refresh empty (%d/%d)", self._consec_fails, KEY_MAX_FAILURES)
                await broadcaster.emit_global({
                    "event": "key:error",
                    "message": f"Refresh failed ({self._consec_fails}/{KEY_MAX_FAILURES})",
                    "time": _now(),
                })
                return None

            except Exception as exc:
                self._consec_fails += 1
                log.exception("Key refresh error (%d/%d): %s",
                              self._consec_fails, KEY_MAX_FAILURES, exc)
                return None
            finally:
                self._usable.set()


# ═══════════════════════════════════════════════════════════════
#  PERCHANCE HTTP CLIENT
# ═══════════════════════════════════════════════════════════════

class PerchanceClient:
    def __init__(self):
        self._s = cloudscraper.create_scraper()
        self._base = API_BASE.rstrip("/")

    def fetch_ad_code(self) -> str:
        try:
            r = self._s.get(f"{self._base}{API_AD_CODE}",
                            timeout=HTTP_TIMEOUT, headers=PERCHANCE_HEADERS)
            r.raise_for_status()
            return r.text.strip()
        except Exception:
            return ""

    def _await_prev(self, key: str) -> None:
        try:
            self._s.get(
                f"{self._base}{API_AWAIT}",
                params={"userKey": key, "__cacheBust": random.random()},
                timeout=20, headers=PERCHANCE_HEADERS,
            )
        except Exception:
            pass

    def generate(self, *, prompt, negative, seed, resolution,
                 guidance, style, user_key, ad_code) -> dict:
        rid = _rid()
        params = {
            "userKey": user_key, "requestId": rid,
            "adAccessCode": ad_code, "__cacheBust": random.random(),
        }
        body = {
            "prompt": prompt, "negativePrompt": negative,
            "seed": seed, "resolution": resolution,
            "guidanceScale": guidance,
            "channel": "ai-text-to-image-generator",
            "subChannel": style,
            "userKey": user_key, "adAccessCode": ad_code,
            "requestId": rid,
        }
        ad_refreshed = False

        for att in range(1, GEN_MAX_RETRIES + 1):
            try:
                r = self._s.post(
                    f"{self._base}{API_GENERATE}",
                    json=body, params=params,
                    timeout=HTTP_TIMEOUT, headers=PERCHANCE_HEADERS,
                )
                r.raise_for_status()
                res = r.json()
            except Exception as exc:
                log.info("Network error attempt %d/%d: %s", att, GEN_MAX_RETRIES, exc)
                time.sleep(1.0)
                continue

            st = res.get("status")

            if st == "success":
                iid  = res.get("imageId")
                urls = res.get("imageDataUrls")
                if iid:
                    return {"imageId": iid, "seed": res.get("seed")}
                if urls:
                    return {"inline": urls[0], "seed": res.get("seed")}
                return {"error": "empty_success"}

            if st == "invalid_key":
                return {"error": "invalid_key"}

            if st == "waiting_for_prev_request_to_finish":
                self._await_prev(user_key)
                time.sleep(0.5)
                continue

            if st == "invalid_ad_access_code" and not ad_refreshed:
                code = self.fetch_ad_code()
                if code:
                    ad_code = code
                    params["adAccessCode"] = code
                    body["adAccessCode"]   = code
                    ad_refreshed = True
                    time.sleep(0.8)
                    continue
                return {"error": "invalid_ad_access_code"}

            if st == "gen_failure" and res.get("type") == 1:
                time.sleep(2.5)
                continue

            if st in (None, "stale_request", "fetch_failure"):
                time.sleep(1.0)
                continue

            log.error("Unhandled status '%s': %.200s", st, str(res))
            return {"error": f"unhandled_{st}"}

        return {"error": "max_retries"}

    def download(self, image_id: str, prefix: str) -> str:
        url = f"{self._base}{API_DOWNLOAD}?imageId={image_id}"
        t0  = time.time()
        bk  = BACKOFF_BASE

        while time.time() - t0 < DOWNLOAD_TIMEOUT:
            try:
                r = self._s.get(url, timeout=HTTP_TIMEOUT,
                                headers=PERCHANCE_HEADERS, stream=True)
                if r.status_code == 200:
                    ct  = r.headers.get("Content-Type", "")
                    ext = ".png" if "png" in ct else ".webp" if "webp" in ct else ".jpg"
                    fn  = _fname(f"{prefix}_{image_id[:12]}{ext}")
                    fp  = str(OUTPUT_DIR / fn)
                    with open(fp, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    return fp
            except Exception:
                pass
            time.sleep(bk)
            bk = min(bk * 1.8, 8.0)

        raise TimeoutError(f"Download timeout ({DOWNLOAD_TIMEOUT}s) for {image_id}")

    def close(self) -> None:
        try:
            self._s.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
#  BROWSER KEY EXTRACTION
# ═══════════════════════════════════════════════════════════════

async def _cdp_mouse(tab, action, x, y, **kw):
    await tab.send(cdp.input_.dispatch_mouse_event(
        type_=action, x=float(x), y=float(y), **kw,
    ))


async def _viewport_centre(tab):
    try:
        v = await tab.evaluate(
            "(()=>({w:innerWidth,h:innerHeight}))()",
            await_promise=False, return_by_value=True,
        )
        return v["w"] / 2, v["h"] / 2
    except Exception:
        return 600.0, 400.0


async def _ls_get(tab, key):
    try:
        return await tab.evaluate(
            f"localStorage&&localStorage.getItem({json.dumps(key)})",
            await_promise=True, return_by_value=True,
        )
    except Exception:
        return None


async def _click_loop(tab, stop: asyncio.Event):
    try:
        await tab.evaluate("window.focus&&window.focus()",
                           await_promise=False, return_by_value=False)
    except Exception:
        pass

    cx, cy = await _viewport_centre(tab)
    last_upd = time.time()

    while not stop.is_set():
        if time.time() - last_upd > 3.0:
            cx, cy = await _viewport_centre(tab)
            last_upd = time.time()

        jx = random.uniform(-CLICK_JITTER, CLICK_JITTER)
        jy = random.uniform(-CLICK_JITTER, CLICK_JITTER)
        x, y = cx + jx, cy + jy

        try:
            await _cdp_mouse(tab, "mouseMoved", x, y, pointer_type="mouse")
            await asyncio.sleep(random.uniform(0.02, 0.08))
            await _cdp_mouse(tab, "mousePressed", x, y,
                             button=cdp.input_.MouseButton.LEFT,
                             click_count=1, buttons=1)
            await asyncio.sleep(random.uniform(0.03, 0.12))
            await _cdp_mouse(tab, "mouseReleased", x, y,
                             button=cdp.input_.MouseButton.LEFT,
                             click_count=1, buttons=0)
        except Exception:
            pass

        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=CLICK_INTERVAL * random.uniform(0.85, 1.15),
            )
            break
        except asyncio.TimeoutError:
            pass


async def _poll_key(tab, stop: asyncio.Event, timeout: int) -> Optional[str]:
    t0 = time.time()
    while not stop.is_set() and (time.time() - t0) < timeout:
        val = await _ls_get(tab, "userKey-0")
        if val:
            return val
        try:
            keys = await tab.evaluate(
                "Object.keys(localStorage||{}).filter(k=>k.includes('userKey'))",
                await_promise=False, return_by_value=True,
            )
            for k in (keys or []):
                v = await _ls_get(tab, k)
                if v:
                    return v
        except Exception:
            pass
        await asyncio.sleep(0.25)
    return None


async def extract_key_via_browser(
    timeout: int = BROWSER_TIMEOUT,
    headless: bool = BROWSER_HEADLESS,
) -> Optional[str]:
    log.info("Browser key extraction  timeout=%ds  headless=%s", timeout, headless)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, partial(_display.ensure, headless))

    try:
        browser = await zd.start(headless=headless)
    except Exception as exc:
        log.exception("Browser start failed: %s", exc)
        return None

    stop   = asyncio.Event()
    result = None

    try:
        main_tab = await browser.get(BROWSER_TARGET)
        await asyncio.sleep(2.0)

        origin_tab = await browser.get(BROWSER_ORIGIN, new_tab=True)
        await asyncio.sleep(1.0)

        await main_tab.bring_to_front()
        await asyncio.sleep(0.5)

        clicker = asyncio.create_task(_click_loop(main_tab, stop))
        poller  = asyncio.create_task(_poll_key(origin_tab, stop, timeout))

        try:
            done, _ = await asyncio.wait({poller}, timeout=timeout)
            if poller in done:
                result = poller.result()
        finally:
            stop.set()
            if not clicker.done():
                clicker.cancel()
                try:
                    await clicker
                except asyncio.CancelledError:
                    pass

        for t in (origin_tab, main_tab):
            try:
                await t.close()
            except Exception:
                pass
    finally:
        try:
            await browser.stop()
        except Exception:
            pass

    if result:
        log.info("Extracted userKey (len=%d)", len(result))
    else:
        log.warning("Key extraction failed within %ds", timeout)
    return result


# ═══════════════════════════════════════════════════════════════
#  GLOBAL SINGLETONS
# ═══════════════════════════════════════════════════════════════

_client   = PerchanceClient()
_executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
_keys     = KeyManager()
_tasks    = TaskStore()
_sse      = Broadcaster()
_queue: Optional[asyncio.Queue] = None

# Track startup readiness (server is up but key may still be loading)
_server_start_time: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════

async def _save_inline(data_url: str, label: str) -> str:
    loop = asyncio.get_running_loop()
    parts = data_url.split(",", 1) if "," in data_url else ["", data_url]
    header, b64 = parts[0], parts[-1]
    ext = ".png" if "png" in header else ".jpg"
    fn  = _fname(f"{label[:30]}_{_ts()}_{_sid()}{ext}")
    fp  = OUTPUT_DIR / fn
    raw = base64.b64decode(b64)
    await loop.run_in_executor(_executor, fp.write_bytes, raw)
    return str(fp)


async def _download_image(image_id: str, label: str) -> str:
    loop   = asyncio.get_running_loop()
    prefix = _fname(f"{label[:30]}_{_ts()}_{_sid()}")
    return await loop.run_in_executor(
        _executor, partial(_client.download, image_id, prefix),
    )


async def _gen_one_image(prompt, task, img_idx, ad_code) -> Optional[dict]:
    loop = asyncio.get_running_loop()
    tid  = task["id"]

    for key_try in range(1, KEY_MAX_RETRIES + 1):
        user_key = await _keys.get()
        if not user_key:
            await _sse.emit(tid, {
                "event": "error",
                "message": "No userKey available — set one via POST /v1/keys/set",
                "time": _now(),
            })
            return None

        result = await loop.run_in_executor(
            _executor,
            partial(
                _client.generate,
                prompt=prompt, negative=task["negative"],
                seed=-1, resolution=task["resolution"],
                guidance=task["guidance"], style=task["style"],
                user_key=user_key, ad_code=ad_code,
            ),
        )

        if result.get("error") == "invalid_key":
            log.warning("invalid_key  task=%s  try=%d/%d", tid[:8], key_try, KEY_MAX_RETRIES)
            await _sse.emit(tid, {
                "event": "key:refresh",
                "message": f"Key invalid — refreshing ({key_try}/{KEY_MAX_RETRIES})",
                "time": _now(),
            })
            new = await _keys.refresh(_sse)
            if new:
                ad_code = await loop.run_in_executor(_executor, _client.fetch_ad_code)
                continue
            return None

        if result.get("error"):
            log.warning("Gen error task=%s: %s", tid[:8], result["error"])
            task["failed_count"] += 1
            await _sse.emit(tid, {
                "event": "error",
                "message": f"Generation error: {result['error']}",
                "prompt": prompt, "index": img_idx, "time": _now(),
            })
            return None

        try:
            if result.get("inline"):
                fpath = await _save_inline(result["inline"], prompt)
            elif result.get("imageId"):
                fpath = await _download_image(result["imageId"], prompt)
            else:
                log.error("Unexpected payload: %s", result)
                return None

            filename  = Path(fpath).name
            image_url = f"/v1/generation/outputs/{filename}"
            seed      = result.get("seed")

            entry = {
                "prompt": prompt, "index": img_idx,
                "image_url": image_url, "seed": seed,
            }
            task["completed"] += 1
            task["results"].append(entry)

            await _sse.emit(tid, {
                "event": "progress", "task_id": tid,
                "prompt": prompt, "index": img_idx,
                "completed": task["completed"], "total": task["total_images"],
                "image_url": image_url, "seed": seed, "time": _now(),
            })

            log.info("✓ %d/%d  task=%s  %s",
                     task["completed"], task["total_images"], tid[:8], filename)
            return entry

        except Exception as exc:
            log.exception("Save/download error task=%s: %s", tid[:8], exc)
            task["failed_count"] += 1
            await _sse.emit(tid, {
                "event": "error", "message": f"Save failed: {exc}",
                "prompt": prompt, "index": img_idx, "time": _now(),
            })
            return None

    return None


async def _run_task(task: dict) -> None:
    loop = asyncio.get_running_loop()
    tid  = task["id"]

    # Wait for key to be available before starting
    if not _keys.is_ready:
        log.info("Task %s waiting for initial key …", tid[:8])
        await _sse.emit(tid, {
            "event": "waiting", "task_id": tid,
            "message": "Waiting for server key initialization …",
            "time": _now(),
        })
        try:
            await _keys.wait_ready(timeout=120)
        except asyncio.TimeoutError:
            task["status"] = "failed"
            task["error"]  = "Key initialization timeout"
            task["finished_at"] = _now()
            await _sse.emit(tid, {
                "event": "failed", "task_id": tid,
                "error": "Key initialization timeout", "time": _now(),
            })
            return

    task["status"]     = "running"
    task["started_at"] = _now()

    await _sse.emit(tid, {
        "event": "started", "task_id": tid,
        "total": task["total_images"], "time": _now(),
    })

    ad_code = await loop.run_in_executor(_executor, _client.fetch_ad_code)

    async def _hb():
        t0 = time.time()
        while task["status"] == "running":
            await asyncio.sleep(SSE_PING_SEC)
            if task["status"] == "running":
                await _sse.emit(tid, {
                    "event": "heartbeat", "task_id": tid,
                    "completed": task["completed"], "total": task["total_images"],
                    "elapsed": round(time.time() - t0, 1), "time": _now(),
                })

    hb = asyncio.create_task(_hb())

    try:
        for prompt in task["prompts"]:
            for i in range(task["count"]):
                await _gen_one_image(prompt, task, i, ad_code)

        if task["completed"] == 0 and task["total_images"] > 0:
            task["status"] = "failed"
            task["error"]  = "No images generated"
        else:
            task["status"] = "completed"

        task["finished_at"] = _now()

        await _sse.emit(tid, {
            "event": task["status"], "task_id": tid,
            "completed": task["completed"], "total": task["total_images"],
            "failed": task["failed_count"], "results": task["results"],
            "error": task.get("error"), "time": _now(),
        })

    except Exception as exc:
        log.exception("Task %s crashed: %s", tid[:8], exc)
        task["status"]      = "failed"
        task["error"]       = str(exc)
        task["finished_at"] = _now()
        await _sse.emit(tid, {
            "event": "failed", "task_id": tid,
            "error": str(exc), "time": _now(),
        })
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        log.info("Task %s  %s  %d/%d  failed=%d",
                 tid[:8], task["status"],
                 task["completed"], task["total_images"], task["failed_count"])


# ═══════════════════════════════════════════════════════════════
#  WORKERS + BACKGROUND JOBS
# ═══════════════════════════════════════════════════════════════

async def _worker(wid: int):
    log.info("Worker-%d online", wid)
    while True:
        job = await _queue.get()
        if job is None:
            _queue.task_done()
            log.info("Worker-%d offline", wid)
            return
        try:
            await _run_task(job["task"])
        except Exception as exc:
            log.exception("Worker-%d unhandled: %s", wid, exc)
        _queue.task_done()


async def _janitor():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            n = _tasks.purge_expired()
            if n:
                log.info("Janitor: purged %d expired tasks (%d remain)", n, _tasks.size)
        except Exception:
            log.exception("Janitor error")


async def _background_key_fetch():
    """
    Fetch userKey via browser IN THE BACKGROUND.
    Server is already accepting connections while this runs.
    """
    skip = os.environ.get("NO_INITIAL_FETCH", "0") in ("1", "true", "True")
    if skip:
        log.info("NO_INITIAL_FETCH → skipping browser key fetch")
        return

    log.info("Background key fetch starting …")
    try:
        key = await extract_key_via_browser(BROWSER_TIMEOUT, BROWSER_HEADLESS)
        if key:
            await _keys.set(key)
            log.info("Background key fetch OK")
        else:
            log.warning("Background key fetch failed — use /v1/keys/set or /v1/keys/refresh")
    except Exception as exc:
        log.exception("Background key fetch error: %s", exc)


# ═══════════════════════════════════════════════════════════════
#  FASTAPI — LIFESPAN
#
#  KEY CHANGE: yield IMMEDIATELY so HF Spaces sees port 7860
#              key fetch runs as a background task AFTER yield
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _queue, _server_start_time

    _server_start_time = time.time()
    _queue = asyncio.Queue(maxsize=QUEUE_CAPACITY)
    loop   = asyncio.get_running_loop()

    # Start Xvfb (fast, synchronous)
    await loop.run_in_executor(None, partial(_display.ensure, BROWSER_HEADLESS))

    # Start workers (instant)
    workers = [asyncio.create_task(_worker(i + 1)) for i in range(WORKER_COUNT)]

    # Start janitor (instant)
    janitor = asyncio.create_task(_janitor())

    # Start key fetch IN BACKGROUND (non-blocking!)
    key_task = asyncio.create_task(_background_key_fetch())

    log.info(
        "═══ SERVER READY ═══  workers=%d  queue_cap=%d  headless=%s  (key fetching in background)",
        WORKER_COUNT, QUEUE_CAPACITY, BROWSER_HEADLESS,
    )

    # ════════════════════════════════════════
    #  yield IMMEDIATELY — port 7860 opens NOW
    #  HF Spaces sees 200 OK on GET /
    #  No more "stuck on Starting"!
    # ════════════════════════════════════════
    yield

    # ──────── shutdown ────────
    log.info("Shutting down …")

    # Cancel background key fetch if still running
    if not key_task.done():
        key_task.cancel()
        try:
            await key_task
        except (asyncio.CancelledError, Exception):
            pass

    for _ in range(WORKER_COUNT):
        await _queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    janitor.cancel()
    try:
        await janitor
    except asyncio.CancelledError:
        pass

    _client.close()
    _executor.shutdown(wait=False)
    await loop.run_in_executor(None, _display.shutdown)
    log.info("═══ SHUTDOWN COMPLETE ═══")


# ═══════════════════════════════════════════════════════════════
#  FASTAPI — APP + ROUTES
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="Perchance Image Generation API", version="3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def _global_err(request: Request, exc: Exception):
    log.exception("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ══════════════════════════════════════
#  ROOT  — HF Spaces readiness check
# ══════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name":    "Perchance Image Generation API",
        "version": "3.0",
        "status":  "running",
        "key_ready": _keys.is_ready,
        "uptime":  round(time.time() - _server_start_time, 1),
        "docs":    "/docs",
        "endpoints": {
            "generate":     "POST /v1/generation/Image/Create",
            "stream":       "GET  /v1/generation/stream/{task_id}",
            "status":       "GET  /v1/generation/status/{task_id}",
            "outputs":      "GET  /v1/generation/outputs/{filename}",
            "health":       "GET  /health",
            "set_key":      "POST /v1/keys/set",
            "refresh_key":  "POST /v1/keys/refresh",
        },
    }


@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "has_key":          _keys.key is not None,
        "key_ready":        _keys.is_ready,
        "queue_size":       _queue.qsize() if _queue else 0,
        "queue_capacity":   QUEUE_CAPACITY,
        "active_tasks":     _tasks.active_count(),
        "stored_tasks":     _tasks.size,
        "sse_subscribers":  _sse.total_subscribers,
        "uptime":           round(time.time() - _server_start_time, 1),
        "time":             _now(),
    }


@app.post("/v1/generation/Image/Create", response_model=GenerationResponse)
async def create_image(req: GenerationRequest):
    prompts = req.prompts if req.prompts else ([req.prompt] if req.prompt else [])
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise HTTPException(400, detail="At least one non-empty prompt is required")

    task = _tasks.create(
        prompts=prompts, count=req.count, resolution=req.resolution,
        guidance=req.guidance_scale, negative=req.negative_prompt, style=req.style,
    )

    try:
        await asyncio.wait_for(_queue.put({"task": task}), timeout=5.0)
    except (asyncio.TimeoutError, asyncio.QueueFull):
        raise HTTPException(503, detail="Server is busy — please retry shortly")

    pos = _queue.qsize()

    await _sse.emit(task["id"], {
        "event": "queued", "task_id": task["id"],
        "position": pos, "total": task["total_images"], "time": _now(),
    })

    return GenerationResponse(
        task_id=task["id"],
        stream=f"/v1/generation/stream/{task['id']}",
        status_url=f"/v1/generation/status/{task['id']}",
        total_images=task["total_images"],
        created_at=task["created_at"],
        queue_pos=pos,
    )


@app.get("/v1/generation/stream/{task_id}")
async def stream_task(request: Request, task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, detail="Task not found")

    sub_q = await _sse.subscribe(task_id)

    async def _stream():
        try:
            yield {
                "event": "connected",
                "data":  json.dumps({
                    "task_id": task_id, "status": task["status"],
                    "total_images": task["total_images"],
                    "completed": task["completed"],
                    "created_at": task["created_at"], "time": _now(),
                }),
            }

            if task["status"] in ("completed", "failed"):
                for r in task["results"]:
                    yield {
                        "event": "progress",
                        "data":  json.dumps({
                            "task_id": task_id, "prompt": r["prompt"],
                            "index": r["index"], "image_url": r["image_url"],
                            "seed": r["seed"], "completed": task["completed"],
                            "total": task["total_images"],
                        }),
                    }
                yield {
                    "event": task["status"],
                    "data":  json.dumps({
                        "task_id": task_id, "completed": task["completed"],
                        "total": task["total_images"],
                        "results": task["results"], "error": task.get("error"),
                    }),
                }
                return

            while True:
                try:
                    ev = await asyncio.wait_for(sub_q.get(), timeout=SSE_PING_SEC + 5)
                except asyncio.TimeoutError:
                    yield {
                        "event": "ping",
                        "data":  json.dumps({"time": _now()}),
                    }
                    if await request.is_disconnected():
                        break
                    if task["status"] in ("completed", "failed"):
                        yield {
                            "event": task["status"],
                            "data":  json.dumps({
                                "task_id": task_id, "completed": task["completed"],
                                "total": task["total_images"],
                                "results": task["results"], "error": task.get("error"),
                            }),
                        }
                        break
                    continue

                etype = ev.get("event", "message")
                yield {"event": etype, "data": json.dumps(ev)}
                if etype in ("completed", "failed"):
                    break
        finally:
            await _sse.unsubscribe(task_id, sub_q)

    return EventSourceResponse(_stream())


@app.get("/v1/generation/status/{task_id}")
async def task_status(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, detail="Task not found")
    return {
        "task_id": task["id"], "status": task["status"],
        "total_images": task["total_images"], "completed": task["completed"],
        "failed_count": task["failed_count"], "created_at": task["created_at"],
        "started_at": task["started_at"], "finished_at": task["finished_at"],
        "results": task["results"], "error": task.get("error"),
    }


@app.get("/v1/generation/outputs/{filename}")
async def serve_image(filename: str):
    fp = OUTPUT_DIR / filename
    if not fp.exists():
        raise HTTPException(404, detail="File not found")
    ext = fp.suffix.lower()
    media = {
        ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    return FileResponse(fp, media_type=media, filename=filename)


@app.post("/v1/keys/set")
async def set_key(body: SetKeyRequest):
    await _keys.set(body.userKey)
    _keys.reset_failures()
    return {"status": "ok", "key_length": len(body.userKey)}


@app.post("/v1/keys/refresh")
async def refresh_key():
    _keys.reset_failures()
    asyncio.create_task(_keys.refresh(_sse))
    return {"status": "started", "message": "Key refresh running in background"}


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    _display.ensure(BROWSER_HEADLESS)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))