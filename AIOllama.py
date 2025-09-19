# ============================================================
# Ollama Super AIO Node for ComfyUI (OpenWebUI+ grade)
# All-in-One: Live + Multi-APIs + KB + Images + Translate + Thinking
# Async + Caching + Profiles + Safe Mode + NSFW toggle (guarded)
# Compatible with DeepSeek / Mistral / Qwen / GPT-OSS / LLaVA / Qwen-VL
# ============================================================

import os
import re
import io
import json
import math
import glob
import time
import base64
import hashlib
import asyncio
import random
import logging
from functools import lru_cache
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
from PIL import Image
from urllib.parse import quote_plus
from collections import Counter, defaultdict
def _ua():
    return {"User-Agent": "ComfyUI-Ollama-SuperAIO/6.0"}

def _clamp(s, limit):
    if not isinstance(s, str):
        try: s = str(s)
        except Exception: return ""
    if limit <= 0: return ""
    if len(s) <= limit: return s
    return s[:limit].rsplit(" ", 1)[0] + "..."

def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _setup_logger(enabled: bool, path: str = ""):
    logger = logging.getLogger("SuperAIO")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    if enabled and path:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def _png_b64_from_ndarray(arr: np.ndarray) -> str:
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        maxv = float(np.max(arr)) if arr.size else 1.0
        if maxv <= 1.01:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _png_b64_from_bytes(img_bytes: bytes) -> str:
    pil = Image.open(io.BytesIO(img_bytes))
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
class OllamaSuperAIO:
    CATEGORY = "AIOllama"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("thinking_output", "result", "sources_json")
    FUNCTION = "run"

    _cached_model_choices = None
    _cached_model_fetch_ts = 0

    def __init__(self):
        self._kb_chunks: List[Dict[str, Any]] = []
        self._kb_idf: Dict[str, float] = {}
        self._kb_ready: bool = False
        self._kb_cache_sig: str = ""
        self._ttl_cache: Dict[str, Tuple[float, Any, int]] = {}
        self._ttl_default: int = 180
        self._safe_mode: bool = True
        self._logger = _setup_logger(False)

    def _model_profile(self, name: str):
        n = (name or "").lower()
        return {
            "is_mistral": "mistral" in n,
            "is_gptoss": "gpt-oss" in n or "oss" in n,
            "is_deepseek": "deepseek" in n,
            "is_qwen": "qwen" in n,
        }
    @classmethod
    def _bootstrap_models_for_dropdown(cls, base="http://127.0.0.1:11434"):
        if cls._cached_model_choices and (time.time() - cls._cached_model_fetch_ts) < 120:
            return cls._cached_model_choices
        try:
            data = requests.get(f"{base.rstrip('/')}/api/tags", headers=_ua(), timeout=2).json()
            models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
            models = models or ["llama3.1", "mistral", "qwen2.5-vl", "deepseek-r1:7b", "gpt-oss", "llava"]
        except Exception:
            models = ["llama3.1", "mistral", "qwen2.5-vl", "deepseek-r1:7b", "gpt-oss", "llava"]
        cls._cached_model_choices = models
        cls._cached_model_fetch_ts = time.time()
        return models

    @classmethod
    def INPUT_TYPES(cls):
        choices = cls._bootstrap_models_for_dropdown()
        return {
            "required": {
                "url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model_choice": (choices, {"default": choices[0] if choices else "llama3.1"}),
                "model_override": ("STRING", {"default": ""}),
                "list_models_on_run": ("BOOLEAN", {"default": True}),
                "refresh_connection": ("BOOLEAN", {"default": False}),
                "keep_alive": ("INT", {"default": 10, "min": -1, "max": 240}),
                "keep_alive_unit": (["minutes", "hours"], {"default": "minutes"}),

                # Async & caching & profiles
                "async_enabled": ("BOOLEAN", {"default": True}),
                "cache_ttl_s": ("INT", {"default": 180, "min": 30, "max": 3600}),
                "profile": (["balanced", "speed", "max_context", "low_latency"], {"default": "balanced"}),
                "log_to_file": ("BOOLEAN", {"default": False}),
                "log_file_path": ("STRING", {"default": ""}),

                # Live search
                "use_live_search": ("BOOLEAN", {"default": True}),
                "wiki_lang": (["en", "nl", "de", "fr"], {"default": "en"}),
                "search_query_override": ("STRING", {"default": ""}),
                "search_max_results": ("INT", {"default": 5, "min": 1, "max": 12}),
                "search_timeout_s": ("INT", {"default": 10, "min": 2, "max": 60}),
                "enhanced_live_fallbacks": ("BOOLEAN", {"default": True}),
                "brave_key": ("STRING", {"default": ""}),
                "serper_key": ("STRING", {"default": ""}),
                "google_cse_key": ("STRING", {"default": ""}),
                "google_cse_cx": ("STRING", {"default": ""}),

                # Multi-API categories (existing + registry)
                "use_weather_apis": ("BOOLEAN", {"default": True}),
                "use_nature_apis": ("BOOLEAN", {"default": True}),
                "use_finance_apis": ("BOOLEAN", {"default": True}),
                "use_news_apis": ("BOOLEAN", {"default": True}),
                "use_media_apis": ("BOOLEAN", {"default": True}),
                "use_general_apis": ("BOOLEAN", {"default": True}),
                "use_scraping_fallback": ("BOOLEAN", {"default": False}),
                "use_animals_apis": ("BOOLEAN", {"default": True}),
                "use_anime_apis": ("BOOLEAN", {"default": True}),
                "use_books_apis": ("BOOLEAN", {"default": True}),
                "use_crypto_apis": ("BOOLEAN", {"default": True}),
                "use_fx_apis": ("BOOLEAN", {"default": True}),
                "use_devfun_apis": ("BOOLEAN", {"default": True}),
                "use_geoip_apis": ("BOOLEAN", {"default": True}),
                "use_music_apis": ("BOOLEAN", {"default": True}),
                "use_sports_apis": ("BOOLEAN", {"default": True}),
                "per_category_limit": ("INT", {"default": 3, "min": 1, "max": 10}),

                # Safe mode & NSFW gating
                "safe_content_mode": ("BOOLEAN", {"default": True}),
                "allow_nsfw_registry": ("BOOLEAN", {"default": False}),

                # Key-based news/web (optional)
                "newsapi_key": ("STRING", {"default": ""}),
                "guardian_key": ("STRING", {"default": ""}),
                "nytimes_key": ("STRING", {"default": ""}),

                # KB / RAG
                "use_knowledge_base": ("BOOLEAN", {"default": True}),
                "knowledge_base_dir": ("STRING", {"default": ""}),
                "kb_max_chunks": ("INT", {"default": 6, "min": 1, "max": 20}),
                "kb_chunk_chars": ("INT", {"default": 900, "min": 300, "max": 2000}),
                "kb_overlap_chars": ("INT", {"default": 120, "min": 0, "max": 800}),

                # Image search
                "image_search_provider": (["off", "duckduckgo", "unsplash", "bing", "pexels"], {"default": "off"}),
                "image_max_results": ("INT", {"default": 4, "min": 1, "max": 12}),
                "unsplash_key": ("STRING", {"default": ""}),
                "bing_key": ("STRING", {"default": ""}),
                "pexels_key": ("STRING", {"default": ""}),

                # Prompting/behavior
                "use_tool_prompt": ("BOOLEAN", {"default": True}),
                "thinking": ("BOOLEAN", {"default": False}),
                "gpt_oss_harmonizer": ("BOOLEAN", {"default": False}),
                "legacy_weather_mode": ("BOOLEAN", {"default": True}),

                # Translate pipeline
                "translate_before_api": ("BOOLEAN", {"default": True}),
                "force_output_english": ("BOOLEAN", {"default": True}),

                # Prompts
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),

                # LLM options
                "temperature": ("FLOAT", {"default": 0.7, "min": -10, "max": 10}),
                "num_predict": ("INT", {"default": 1024, "min": -1, "max": 8192}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 400}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0}),
                "max_context_chars": ("INT", {"default": 12000, "min": 1200, "max": 700000}),

                # Extra
                "context_messages_json": ("STRING", {"multiline": True, "default": ""}),
                "api_timeout_override_s": ("INT", {"default": 0, "min": 0, "max": 300}),
                "detect_multimodal": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "optional_image_input": ("IMAGE", {"default": None}),
                "images_b64_json": ("STRING", {"multiline": True, "default": ""}),
                "optional_image_b64": ("STRING", {"multiline": True, "default": ""}),
                "batch_image_paths": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }
    def _safe_bool(self, v, default=False):
        try:
            if isinstance(v, bool): return v
            if isinstance(v, str): return v.strip().lower() in ("1","true","yes","y","on")
            if isinstance(v, (int, float)): return v != 0
        except Exception: pass
        return default

    def _parse_json_safe(self, s: str, default, dbg_label: str):
        if not s or not s.strip(): return default
        try: return json.loads(s)
        except Exception as e:
            self._logger.debug(f"[AIO] JSON parse failed for {dbg_label}: {e}")
            return default

    def _health_check(self, base_url: str) -> bool:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/api/tags", headers=_ua(), timeout=6)
            r.raise_for_status()
            return True
        except Exception as e:
            self._logger.debug(f"[AIO] Health check failed: {e}")
            return False

    def _with_retry(self, func, tries=2, delay=0.35, backoff=1.6, jitter=0.1, *args, **kwargs):
        last = None
        cur = delay
        for _ in range(max(1, tries)):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last = e
                time.sleep(cur + random.uniform(0, jitter))
                cur *= backoff
        raise last

    def _cache_get(self, key):
        rec = self._ttl_cache.get(key)
        if not rec: return None
        ts, val, ttl = rec
        if time.time() - ts > ttl:
            self._ttl_cache.pop(key, None)
            return None
        return val

    def _cache_put(self, key, value, ttl=None):
        self._ttl_cache[key] = (time.time(), value, ttl or self._ttl_default)

    def _cache_key(self, name, params):
        try:
            return name + "::" + hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        except Exception:
            return name + "::" + str(params)
    def _list_ollama_models(self, base_url, timeout=6):
        try:
            data = requests.get(f"{base_url}/api/tags", headers=_ua(), timeout=timeout).json()
            return [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
        except Exception:
            return []

    def _show_ollama_model(self, base_url, model, timeout=6):
        try:
            return requests.get(f"{base_url}/api/show", params={"name": model}, headers=_ua(), timeout=timeout).json()
        except Exception:
            return None

    def _is_probably_multimodal(self, base_url, model, allow_probe=True):
        name = (model or "").lower()
        if any(k in name for k in ["-vl"," vision","vision-","vl-","mm","multimodal","llava","qwen-vl","minicpm-v"]):
            return True
        if not allow_probe: return False
        info = self._show_ollama_model(base_url, model)
        if not info: return False
        js = json.dumps(info).lower()
        return any(tag in js for tag in ["vision","images","mmproj","clip","q_former"])

    def _inject_toolprompt(self, system_prompt: str, enabled: bool, thinking: bool):
        if not enabled: return system_prompt
        extra = "[TOOLPROMPT] You may use live search and public APIs to retrieve up-to-date info. Cite sources as [n]."
        if thinking:
            extra += "\nWhen reasoning, keep <think> concise and provide the final answer in <final>."
        return (extra + "\n" + (system_prompt or "")).strip()

    def _harmonize_for_gptoss(self, model_name: str, messages: list, enabled: bool):
        if not enabled: return messages
        name = (model_name or "").lower()
        if "gpt-oss" not in name: return messages
        sys = "\n".join([m["content"] for m in messages if m.get("role") == "system"]).strip()
        usr = "\n".join([m["content"] for m in messages if m.get("role") == "user"]).strip()
        return [{"role": "system", "content": f"<|start|><|system|>{sys or 'You are a helpful assistant.'}<|user|>{usr}<|assistant|>"}]
    def _detect_language_simple(self, text):
        s = (text or "").strip()
        if not s: return "en"
        if re.search(r"[àáâäèéêëìíîïòóôöùúûüçñß]", s, flags=re.I) or re.search(r"\b(het|de|een|weer|vandaag|morgen|nieuws)\b", s, flags=re.I):
            return "nl"
        return "en"

    def _libre_detect(self, text, timeout=6):
        try:
            r = requests.post("https://libretranslate.com/detect", data={"q": text},
                              timeout=timeout, headers={**_ua(), "Accept":"application/json"})
            r.raise_for_status()
            arr = r.json()
            if isinstance(arr, list) and arr:
                return arr[0].get("language", "en")
        except Exception: pass
        return None

    def _translate_deepl(self, text, target="EN", timeout=8):
        api_key = os.environ.get("DEEPL_API_KEY","").strip()
        if not api_key or not text: return None
        try:
            r = requests.post("https://api-free.deepl.com/v2/translate",
                              data={"auth_key": api_key, "text": text, "target_lang": target.upper()},
                              timeout=timeout, headers=_ua())
            r.raise_for_status()
            data = r.json()
            trs = data.get("translations", [])
            if trs: return trs[0].get("text")
        except Exception: return None
        return None

    def _translate_libre(self, text, target="en", timeout=8):
        try:
            r = requests.post("https://libretranslate.com/translate",
                              data={"q": text, "source":"auto","target": target, "format":"text"},
                              timeout=timeout, headers={**_ua(),"Accept":"application/json"})
            r.raise_for_status()
            return r.json().get("translatedText")
        except Exception: return None

    def _translate_lingva(self, text, target="en", source="auto", timeout=8):
        try:
            src = source if source != "auto" else "auto"
            url = f"https://lingva.ml/api/v1/{src}/{target}/{quote_plus(text)}"
            r = requests.get(url, headers=_ua(), timeout=timeout)
            r.raise_for_status()
            return r.json().get("translation")
        except Exception: return None

    def _translate_mymemory(self, text, target="en", timeout=8):
        try:
            r = requests.get("https://api.mymemory.translated.net/get",
                             params={"q": text, "langpair": f"auto|{target}"}, headers=_ua(), timeout=timeout)
            r.raise_for_status()
            return r.json().get("responseData", {}).get("translatedText")
        except Exception: return None

    def _translate_if_needed(self, text, target="en", enable=True):
        if not enable: return text, None, False
        src_guess = self._libre_detect(text) or self._detect_language_simple(text)
        if (src_guess or "en").lower().startswith(target.lower()):
            return text, (src_guess or "en"), False
        out = self._translate_deepl(text, target=target) \
              or self._translate_libre(text, target=target) \
              or self._translate_lingva(text, target=target, source="auto") \
              or self._translate_mymemory(text, target=target)
        return (out or text), (src_guess or "auto"), bool(out)
    def _duckduckgo_instant(self, query, timeout=10):
        try:
            r = requests.get("https://api.duckduckgo.com/",
                             params={"q": query, "format":"json", "no_redirect":"1", "no_html":"1"},
                             headers=_ua(), timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception: return None

    def _wiki_opensearch(self, query, limit=3, timeout=10, lang="en"):
        try:
            r = requests.get(f"https://{lang}.wikipedia.org/w/api.php",
                             params={"action":"opensearch","search": query, "limit": str(limit), "format":"json"},
                             headers=_ua(), timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception: return None

    def _wiki_summary(self, title, timeout=10, lang="en"):
        try:
            safe = quote_plus(title.replace(" ", "_"))
            r = requests.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{safe}",
                             headers=_ua(), timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return {
                "title": data.get("title") or title,
                "extract": data.get("extract") or "",
                "url": data.get("content_urls",{}).get("desktop",{}).get("page")
                       or f"https://{lang}.wikipedia.org/wiki/{safe}"
            }
        except Exception: return None

    def _search_brave(self, query, key, max_results=5, timeout=8):
        if not key: return [], []
        try:
            r = requests.get("https://api.search.brave.com/res/v1/web/search",
                             params={"q": query, "count": max_results},
                             headers={**_ua(), "Accept":"application/json", "X-Subscription-Token": key},
                             timeout=timeout)
            r.raise_for_status()
            data = r.json()
            snips, srcs = [], []
            for it in (data.get("web",{}).get("results",[]) or [])[:max_results]:
                title = it.get("title") or ""
                desc  = it.get("description") or ""
                url   = it.get("url") or ""
                if desc: snips.append(f"{title}: {desc}" if title else desc)
                if url:  srcs.append({"type":"brave","title": title or url,"url": url})
            return snips, srcs
        except Exception: return [], []

    def _search_serper(self, query, key, max_results=5, timeout=8):
        if not key: return [], []
        try:
            r = requests.post("https://google.serper.dev/search",
                              json={"q": query, "num": max_results},
                              headers={**_ua(), "X-API-KEY": key, "Content-Type":"application/json"},
                              timeout=timeout)
            r.raise_for_status()
            data = r.json()
            snips, srcs = [], []
            for it in (data.get("organic",[]) or [])[:max_results]:
                title = it.get("title") or ""
                desc  = it.get("snippet") or ""
                url   = it.get("link") or ""
                if desc: snips.append(f"{title}: {desc}" if title else desc)
                if url:  srcs.append({"type":"serper","title": title or url,"url": url})
            return snips, srcs
        except Exception: return [], []

    def _search_google_cse(self, query, key, cx, max_results=5, timeout=8):
        if not key or not cx: return [], []
        try:
            r = requests.get("https://www.googleapis.com/customsearch/v1",
                             params={"q": query, "key": key, "cx": cx, "num": max_results},
                             headers=_ua(), timeout=timeout)
            r.raise_for_status()
            data = r.json()
            snips, srcs = [], []
            for it in (data.get("items",[]) or [])[:max_results]:
                title = it.get("title") or ""
                desc  = it.get("snippet") or ""
                url   = it.get("link") or ""
                if desc: snips.append(f"{title}: {desc}" if title else desc)
                if url:  srcs.append({"type":"google_cse","title": title or url,"url": url})
            return snips, srcs
        except Exception: return [], []

    def _rank_live_snippet(self, txt):
        if not txt: return 0
        score = min(len(txt), 800)
        for t in ["today","year","update","announced","nieuw","vandaag","breaking","recent"]:
            if t in txt.lower(): score += 150
        return score

    def _collect_live_once(self, query, max_results=5, timeout=10, wiki_lang="en"):
        sources, snippets = [], []
        ddg = None if query.strip()=="" else self._duckduckgo_instant(query, timeout=timeout)
        if ddg:
            abs_txt = (ddg.get("AbstractText") or "").strip()
            abs_url = (ddg.get("AbstractURL") or "").strip()
            if abs_txt: snippets.append(abs_txt)
            if abs_url: sources.append({"type":"duckduckgo","title":"Instant Answer","url": abs_url})
            for rt in (ddg.get("RelatedTopics") or [])[:max_results]:
                if isinstance(rt, dict):
                    txt = (rt.get("Text") or "").strip()
                    url = rt.get("FirstURL")
                    if txt: snippets.append(txt)
                    if url: sources.append({"type":"duckduckgo","title":"Related","url": url})
        wiki = self._wiki_opensearch(query, limit=max_results, timeout=timeout, lang=wiki_lang)
        if isinstance(wiki, list) and len(wiki) >= 4:
            titles, urls = wiki[1] or [], wiki[3] or []
            for i, t in enumerate(titles[:max_results]):
                s = self._wiki_summary(t, timeout=timeout, lang=wiki_lang)
                if s and s.get("extract"):
                    snippets.append(f"{s['title']}: {s['extract']}")
                    sources.append({"type":"wikipedia","title": s.get("title") or t,"url": s.get("url") or (urls[i] if i < len(urls) else None)})
        # dedup and rank
        seen, dedup_sources = set(), []
        for s in sources:
            u = s.get("url")
            if not u or u in seen: continue
            dedup_sources.append(s); seen.add(u)
        sn_sorted = sorted((x for x in snippets if x.strip()), key=self._rank_live_snippet, reverse=True)
        return sn_sorted, dedup_sources

    def _collect_live_enhanced(self, query, max_results=5, timeout=10, wiki_lang="en",
                               brave_key="", serper_key="", google_cse_key="", google_cse_cx=""):
        snips, srcs = self._collect_live_once(query, max_results=max_results, timeout=timeout, wiki_lang=wiki_lang)
        if snips and srcs: return snips, srcs
        extra_snips, extra_srcs = [], []
        s1, c1 = self._search_brave(query, key=brave_key, max_results=max_results, timeout=timeout); extra_snips += s1; extra_srcs += c1
        s2, c2 = self._search_serper(query, key=serper_key, max_results=max_results, timeout=timeout); extra_snips += s2; extra_srcs += c2
        s3, c3 = self._search_google_cse(query, key=google_cse_key, cx=google_cse_cx, max_results=max_results, timeout=timeout); extra_snips += s3; extra_srcs += c3
        extra_snips = [x for x in extra_snips if x and x.strip()]
        seen, dedup = set(), []
        for s in extra_srcs:
            u = s.get("url")
            if not u or u in seen: continue
            dedup.append(s); seen.add(u)
        return (snips or extra_snips), (srcs or dedup)
    def _extract_location(self, text: str):
        if not text: return None
        m = re.search(r"\b(in|near|bij|rond)\s+([A-Za-z\u00C0-\u024F\-\s']{2,})\b", text, flags=re.I)
        if m: return m.group(2).strip()
        caps = re.findall(r"\b([A-Z][a-z\u00C0-\u024F\-']{2,}(?:\s+[A-Z][a-z\u00C0-\u024F\-']{2,}){0,2})\b", text)
        if caps: return caps[0].strip()
        return None

    def _normalize_place(self, place: str):
        if not place: return ""
        s = place.strip()
        fixes = {
            "nijeeveen": "Nijeveen",
            "nijeveeen": "Nijeveen",
            "nijeven": "Nijeveen",
            "nijeveen, nl": "Nijeveen, Drenthe, Netherlands",
            "nijeveen, nederland": "Nijeveen, Drenthe, Netherlands",
        }
        key = s.lower()
        if key in fixes: return fixes[key]
        if "nijeveen" in key and "nether" not in key and "nederland" not in key and "drenthe" not in key:
            return "Nijeveen, Drenthe, Netherlands"
        return s

    def _weather_wttr_legacy(self, query, timeout=10):
        try:
            loc = (query or "").strip()
            if not loc: return None
            url = f"https://wttr.in/{quote_plus(loc)}?format=3"
            r = requests.get(url, headers=_ua(), timeout=timeout)
            r.raise_for_status()
            return f"wttr.in: {r.text.strip()}"
        except Exception:
            return None

    def _weather_open_meteo(self, query, timeout=8):
        try:
            loc = self._normalize_place(self._extract_location(query) or query)
            rg = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                              params={"name": loc, "count": 1}, headers=_ua(), timeout=timeout)
            rg.raise_for_status()
            geo = rg.json()
            if not geo.get("results"):
                alt = f"{loc}, Netherlands" if "nether" not in loc.lower() else loc
                rg = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                                  params={"name": alt, "count": 1}, headers=_ua(), timeout=timeout)
                rg.raise_for_status()
                geo = rg.json()
                if not geo.get("results"):
                    return None
            lat = geo["results"][0]["latitude"]; lon = geo["results"][0]["longitude"]
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                             params={"latitude": lat, "longitude": lon, "current_weather":"true", "timezone":"auto"},
                             headers=_ua(), timeout=timeout)
            r.raise_for_status()
            data = r.json().get("current_weather", {})
            return f"Open-Meteo: {data.get('temperature')}°C, wind {data.get('windspeed')} km/h, code {data.get('weathercode')}"
        except Exception: return None

    def _weather_openaq(self, query, timeout=8):
        try:
            city = self._normalize_place(self._extract_location(query) or query)
            r = requests.get("https://api.openaq.org/v2/latest",
                             params={"limit":1,"page":1,"offset":0,"sort":"desc","radius":10000,"city": city},
                             headers=_ua(), timeout=timeout)
            r.raise_for_status()
            m = r.json().get("results", [{}])[0].get("measurements", [])
            if not m: return None
            out = ", ".join(f"{x['parameter']}: {x['value']} {x['unit']}" for x in m[:3])
            return f"OpenAQ air quality: {out}"
        except Exception: return None

    def _weather_with_fallbacks(self, query, timeout=10):
        for fn in [self._weather_wttr_legacy, self._weather_open_meteo, self._weather_openaq]:
            w = fn(query, timeout=timeout)
            if w: return w
        loc = self._normalize_place(query)
        return self._weather_wttr_legacy(loc, timeout=timeout)
    def _looks_nsfw_url(self, url: str) -> bool:
        if not isinstance(url, str): return False
        u = url.lower()
        nsfw_markers = ["/nsfw/","nsfw_","hentai","rule34","xbooru","gelbooru","booru","porn","adult","erotic"]
        return any(m in u for m in nsfw_markers)

    def _safe_allow(self, name: str, url: str) -> bool:
        if not self._safe_mode: return True
        if self._looks_nsfw_url(url or ""): return False
        bad_names = ["rule34","danbooru","xbooru","gelbooru","nekobot"]
        n = (name or "").lower()
        if any(b in n for b in bad_names): return False
        return True

    def _guess_best_fields(self, obj):
        if not isinstance(obj, dict): return str(obj)
        candidates = []
        for k in ["fact","text","quote","description","extract","title","setup","answer","name","message","url","image","link","summary"]:
            v = obj.get(k)
            if isinstance(v, (str,int,float)) and str(v).strip():
                candidates.append(f"{k}: {v}")
        if candidates:
            return " | ".join(candidates[:3])
        try:
            s = json.dumps({k: obj[k] for k in list(obj)[:6]}, ensure_ascii=False)
            return _clamp(s, 300)
        except Exception:
            return _clamp(str(obj), 300)

    def _json_first(self, data):
        if isinstance(data, dict):
            return self._guess_best_fields(data)
        if isinstance(data, list) and data:
            return self._guess_best_fields(data[0])
        return None

    def _extract_media_source(self, data):
        if isinstance(data, dict):
            for k in ["image","url","link","thumbnail","message","file","fileUrl"]:
                v = data.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    return v
        if isinstance(data, list):
            for it in data:
                s = self._extract_media_source(it)
                if s: return s
        return None

    def _fetch_generic(self, url, timeout=8, headers=None, params=None):
        r = requests.get(url, headers=headers or _ua(), params=params, timeout=timeout)
        r.raise_for_status()
        ctype = r.headers.get("content-type","").lower()
        if "application/json" in ctype or r.text.strip().startswith(("{","[")):
            try:
                return r.json()
            except Exception:
                pass
        try:
            return r.json()
        except Exception:
            return {"_raw": _clamp(r.text, 500)}

    def _fetch_json(self, url, timeout=8, headers=None, params=None):
        return self._fetch_generic(url, timeout=timeout, headers=headers, params=params)

    def _fetch_xml(self, url, timeout=8, headers=None, params=None):
        import xml.etree.ElementTree as ET
        r = requests.get(url, headers=headers or _ua(), params=params, timeout=timeout)
        r.raise_for_status()
        try:
            root = ET.fromstring(r.text)
            return {root.tag: [child.tag for child in root[:10]]}
        except Exception:
            return {"_raw": _clamp(r.text, 500)}

    def _registry_items(self, category, query):
        REG = {
            "animals": [
                {"name":"CatFact","url":"https://catfact.ninja/fact"},
                {"name":"DogFact","url":"https://dog-api.kinduff.com/api/facts"},
                {"name":"DogCEO","url":"https://dog.ceo/api/breeds/image/random","media":True},
                {"name":"RandomDog","url":"https://random.dog/woof.json","media":True},
                {"name":"MeowFacts","url":"https://meowfacts.herokuapp.com/"},
                {"name":"RandomFox","url":"https://randomfox.ca/floof/","media":True},
                {"name":"RandomDuck","url":"https://random-d.uk/api/v2/random","media":True},
                {"name":"Shibe","url":"https://shibe.online/api/shibes","media":True},
                {"name":"Elephant","url":"https://elephant-api.herokuapp.com/elephants/random"},
                {"name":"Xeno-canto","url":"https://xeno-canto.org/api/2/recordings?query=bird"},
                {"name":"HTTP Cat 100","url":"https://http.cat/100","media":True},
                {"name":"HTTP Dog 100","url":"https://http.dog/100","media":True},
                {"name":"Placebear","url":"https://placebear.com/400/300","media":True},
                {"name":"Place-puppy","url":"https://place-puppy.com/400x300","media":True},
            ],
            "anime_sfw": [
                {"name":"NekosBest (SFW)","url":"https://nekos.best/api/v2/nekos","media":True},
                {"name":"Waifu.pics (SFW)","url":"https://waifu.pics/api/sfw/waifu","media":True},
                {"name":"Waifu.im (random)","url":"https://waifu.im/api/random","media":True},
            ],
            "books": [
                {"name":"OpenLibrary cover","url":"https://covers.openlibrary.org/b/id/240727-S.jpg","media":True},
                {"name":"Gutendex","url":"https://gutendex.com/books/"},
                {"name":"Google Books HP","url":"https://www.googleapis.com/books/v1/volumes?q=harry+potter"},
                {"name":"PoetryDB","url":"https://poetrydb.org/title/Poem"},
                {"name":"QuoteGarden","url":"https://quote-garden.herokuapp.com/api/v3/quotes/random"},
                {"name":"Favqs qotd","url":"https://favqs.com/api/qotd"},
            ],
            "crypto": [
                {"name":"CoinGecko markets","url":"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc"},
                {"name":"CoinDesk","url":"https://api.coindesk.com/v1/bpi/currentprice.json"},
                {"name":"Coincap","url":"https://api.coincap.io/v2/assets"},
                {"name":"Binance ticker","url":"https://api.binance.com/api/v3/ticker/price"},
                {"name":"Kraken XBTUSD","url":"https://api.kraken.com/0/public/Ticker?pair=XBTUSD"},
            ],
            "fx": [
                {"name":"Frankfurter","url":"https://api.frankfurter.app/latest"},
                {"name":"Exchangerate.host","url":"https://api.exchangerate.host/latest"},
                {"name":"Open ER API USD","url":"https://open.er-api.com/v6/latest/USD"},
                {"name":"ECB XML (daily)","url":"https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml","xml":True},
            ],
            "devfun": [
                {"name":"JokeAPI","url":"https://v2.jokeapi.dev/joke/Any"},
                {"name":"BoredAPI","url":"https://www.boredapi.com/api/activity/"},
                {"name":"AdviceSlip","url":"https://api.adviceslip.com/advice"},
                {"name":"Numbers trivia","url":"http://numbersapi.com/random/trivia"},
                {"name":"RandomUser","url":"https://randomuser.me/api/"},
                {"name":"YesNo","url":"https://yesno.wtf/api"},
                {"name":"Kanye","url":"https://api.kanye.rest/"},
                {"name":"OpenTDB 3","url":"https://opentdb.com/api.php?amount=3"},
            ],
            "geoip": [
                {"name":"ipify","url":"https://api.ipify.org?format=json"},
                {"name":"ipapi.co","url":"https://ipapi.co/json/"},
                {"name":"ip-api","url":"http://ip-api.com/json/"},
                {"name":"ipinfo.io","url":"https://ipinfo.io/json"},
                {"name":"geojs","url":"https://get.geojs.io/v1/ip/geo.json"},
            ],
            "music": [
                {"name":"MusicBrainz (Queen)","url":"https://musicbrainz.org/ws/2/artist?query=queen&fmt=json"},
                {"name":"iTunes search","url":"https://itunes.apple.com/search?term=jack+johnson&limit=3"},
                {"name":"Lyrics.ovh","url":"https://api.lyrics.ovh/v1/Coldplay/Yellow"},
                {"name":"TheAudioDB","url":"https://www.theaudiodb.com/api/v1/json/1/search.php?s=coldplay"},
                {"name":"Mixcloud popular","url":"https://api.mixcloud.com/popular/"},
            ],
            "sports": [
                {"name":"PokeAPI #1","url":"https://pokeapi.co/api/v2/pokemon/1/"},
                {"name":"Chess.com Magnus","url":"https://api.chess.com/pub/player/magnuscarlsen"},
                {"name":"RAWG sample","url":"https://api.rawg.io/api/games"},
                {"name":"Freetogame","url":"https://www.freetogame.com/api/games"},
                {"name":"MLB teams","url":"https://statsapi.mlb.com/api/v1/teams"},
                {"name":"FPL bootstrap","url":"https://fantasy.premierleague.com/api/bootstrap-static/"},
                {"name":"ESPN sports","url":"https://api.espn.com/v1/sports"},
            ],
            "images_sfw": [
                {"name":"Shibe","url":"https://shibe.online/api/shibes","media":True},
                {"name":"RandomFox","url":"https://randomfox.ca/floof/","media":True},
                {"name":"RandomDuck","url":"https://random.d.uk/api/v2/random","media":True},  # tolerant fallback
            ],
            "weather_bulk": [
                {"name":"wttr global","url":"https://wttr.in/?format=j1"},
                {"name":"met.no sample","url":"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=60.10&lon=9.58"},
                {"name":"weather.gov sample","url":"https://api.weather.gov/gridpoints/MPX/107,71/forecast"},
                {"name":"open-meteo hourly","url":"https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"},
            ],
            # NSFW registry would go here only if allow_nsfw_registry=True (not included by default).
        }
        return REG.get(category, [])
    def _run_registry_category(self, category, query, limit=3, timeout=8):
        items = self._registry_items(category, query)
        out_snips, out_sources = [], []
        hits = 0
        for it in items:
            if hits >= limit: break
            url = it["url"]
            if not self._safe_allow(it["name"], url):  # safe mode guard
                continue
            key = self._cache_key(f"reg::{category}", {"url": url, "q": query})
            cached = self._cache_get(key)
            if cached is not None:
                data = cached
            else:
                try:
                    data = self._fetch_xml(url, timeout=timeout) if it.get("xml") else self._fetch_json(url, timeout=timeout)
                    self._cache_put(key, data, ttl=180)
                except Exception:
                    continue
            media_url = self._extract_media_source(data)
            if self._safe_mode and self._looks_nsfw_url(media_url or ""):
                continue
            if it.get("media", False):
                m = media_url or (isinstance(data, str) and data if str(data).startswith("http") else None)
                if m and isinstance(m, str):
                    out_sources.append({"type": category, "title": it["name"], "url": m})
                    out_snips.append(f"{it['name']}: {m}")
                    hits += 1
                    continue
            s = self._json_first(data)
            if s:
                out_snips.append(f"{it['name']}: {s}")
                if isinstance(media_url, str) and media_url.startswith("http"):
                    out_sources.append({"type": category, "title": it["name"], "url": media_url})
                hits += 1
        return out_snips, out_sources

    def _collect_registry_bulk(self, query, cfg):
        categories = []
        if cfg.get("use_animals_apis"): categories.append("animals")
        if cfg.get("use_anime_apis"): categories.append("anime_sfw")
        if cfg.get("use_books_apis"): categories.append("books")
        if cfg.get("use_crypto_apis"): categories.append("crypto")
        if cfg.get("use_fx_apis"): categories.append("fx")
        if cfg.get("use_devfun_apis"): categories.append("devfun")
        if cfg.get("use_geoip_apis"): categories.append("geoip")
        if cfg.get("use_music_apis"): categories.append("music")
        if cfg.get("use_sports_apis"): categories.append("sports")
        if cfg.get("use_media_apis"): categories.append("images_sfw")
        categories.append("weather_bulk")

        total_snips, total_srcs = [], []
        for cat in categories:
            sn, sc = self._run_registry_category(
                category=cat, query=query,
                limit=cfg.get("per_category_limit", 3),
                timeout=cfg.get("search_timeout_s", 10),
            )
            total_snips += sn
            total_srcs += sc

        # Cross-reference: if live sources domains appear in registry sources, boost ordering later (ContextBuilder orders by appearance)
        return total_snips, total_srcs
    def _kb_signature(self, kb_dir, chunk, overlap):
        items = []
        for path in sorted(glob.glob(os.path.join(kb_dir, "**/*.txt"), recursive=True) +
                           glob.glob(os.path.join(kb_dir, "**/*.md"), recursive=True)):
            try:
                st = os.stat(path)
                items.append(f"{path}|{int(st.st_mtime)}|{st.st_size}")
            except Exception:
                continue
        sig_str = "|".join(items) + f"|{chunk}|{overlap}"
        return hashlib.sha256(sig_str.encode("utf-8")).hexdigest()

    def _read_textfile(self, path):
        try:
            with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception: return ""

    def _chunk_text(self, text, chunk_size=900, overlap=120):
        text = re.sub(r"\s+", " ", text).strip()
        if not text: return []
        chunks, i, n = [], 0, len(text)
        while i < n:
            end = min(i + chunk_size, n)
            seg = text[i:end]
            m = re.search(r".*[\.!?…](\s|$)", seg)
            if m and (i + m.end()) - i >= chunk_size * 0.6:
                end = i + m.end(); seg = text[i:end]
            chunks.append(seg)
            if end >= n: break
            i = max(end - overlap, i + 1)
        return chunks

    def _tokenize(self, s):
        return [t.lower() for t in re.findall(r"[a-zA-Z0-9\u00C0-\u024F]+", s)]

    def _build_kb_index(self, kb_dir, chunk_chars, overlap_chars):
        files = sorted(glob.glob(os.path.join(kb_dir, "**/*.txt"), recursive=True) +
                       glob.glob(os.path.join(kb_dir, "**/*.md"), recursive=True))
        chunks = []
        for path in files:
            txt = self._read_textfile(path)
            if not txt: continue
            title = os.path.basename(path)
            for c in self._chunk_text(txt, chunk_size=chunk_chars, overlap=overlap_chars):
                toks = self._tokenize(c)
                chunks.append({"title": title,"path": path,"text": c,"tf": Counter(toks),"len": len(toks)})
        df = defaultdict(int)
        for ch in chunks:
            for w in set(ch["tf"].keys()):
                df[w] += 1
        N = max(1, len(chunks))
        idf = {w: math.log((N + 1) / (dfw + 1)) + 1.0 for w, dfw in df.items()}
        self._kb_chunks, self._kb_idf, self._kb_ready = chunks, idf, True

    def _ensure_kb(self, kb_dir, chunk_chars, overlap_chars):
        if not kb_dir or not os.path.isdir(kb_dir):
            self._kb_ready = False; return
        sig = self._kb_signature(kb_dir, chunk_chars, overlap_chars)
        if getattr(self, "_kb_cache_sig", None) != sig:
            self._kb_cache_sig = sig
            self._build_kb_index(kb_dir, chunk_chars, overlap_chars)

    def _kb_search(self, query, k=6):
        if not getattr(self, "_kb_ready", False) or not getattr(self, "_kb_chunks", []):
            return []
        q_toks = self._tokenize(query)
        if not q_toks: return []
        q_tf = Counter(q_toks)

        def dot(tf1, tf2, idf):
            s = 0.0
            for w, c in tf1.items():
                if w in tf2:
                    s += (c * idf.get(w, 1.0)) * (tf2[w] * idf.get(w, 1.0))
            return s

        def norm(tf, idf):
            s = 0.0
            for w, c in tf.items():
                v = c * idf.get(w, 1.0); s += v * v
            return math.sqrt(s) if s > 0 else 1.0

        qn = norm(q_tf, self._kb_idf)
        scored = []
        for ch in self._kb_chunks:
            d = dot(q_tf, ch["tf"], self._kb_idf)
            sim = d / (qn * norm(ch["tf"], self._kb_idf))
            sim += 0.05 * (1.0 / (1.0 + max(0, ch["len"] - 250) / 250.0))
            scored.append((sim, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    def _estimate_context_budgets(self, max_context_chars: int):
        total = max(4000, int(max_context_chars))
        if total <= 20000:
            ratios = {"kb": 0.40, "live": 0.30, "api": 0.18, "images": 0.06, "userimg": 0.06}
        elif total <= 160000:
            ratios = {"kb": 0.36, "live": 0.30, "api": 0.20, "images": 0.07, "userimg": 0.07}
        else:
            ratios = {"kb": 0.32, "live": 0.30, "api": 0.24, "images": 0.07, "userimg": 0.07}
        budgets = {k: int(total * v) for k, v in ratios.items()}
        diff = total - sum(budgets.values())
        if diff != 0: budgets["api"] += diff
        return budgets

    def _build_context_v3(self, kb_hits, live_snips, api_snips, live_sources, image_items, user_images_info,
                          max_context_chars=3600):
        budgets = self._estimate_context_budgets(max_context_chars)
        live_sources = live_sources or []
        image_items = image_items or []

        # Safe mode filter
        if self._safe_mode:
            def ok(src):
                return self._safe_allow(src.get("title",""), src.get("url") or "")
            live_sources = [s for s in live_sources if ok(s)]
            image_items = [i for i in image_items if self._safe_allow(i.get("title",""), i.get("url") or i.get("image") or "")]

        def take_with_budget(items, budget, clamp_len):
            buf, used = [], 0
            for it in items:
                s = _clamp(str(it), clamp_len); extra = len(s) + 2
                if used + extra > budget: break
                buf.append(s); used += extra
            return buf

        kb_lines = []
        for ch in (kb_hits or []):
            title = ch.get("title") or ch.get("path") or "KB"
            text  = ch.get("text") or ""
            kb_lines.append(f"{title}: {_clamp(text, 700)}")
        kb_take  = take_with_budget(kb_lines, budgets["kb"], 700)
        live_take = take_with_budget([s for s in (live_snips or []) if s.strip()], budgets["live"], 600)
        api_take  = take_with_budget([s for s in (api_snips or []) if s.strip()], budgets["api"], 650)

        img_lines = []
        for it in image_items:
            cap = it.get("title") or "Image"
            url = it.get("url") or it.get("image") or ""
            img_lines.append(_clamp(f"{cap} — {url}", 220))
        img_take = take_with_budget(img_lines, budgets["images"], 220)

        user_lines = []
        if user_images_info:
            user_lines.append(f"User provided images: {len(user_images_info)} item(s).")
            for i, it in enumerate(user_images_info[:8], 1):
                user_lines.append(f"- Image {i}: {_clamp(str(it), 160)}")
        user_take = take_with_budget(user_lines, budgets["userimg"], 200)

        all_sources = []
        for ch in (kb_hits or []):
            all_sources.append({"type":"kb","title": ch.get("title") or "KB","url": ch.get("path")})
        for s in live_sources:
            if isinstance(s, dict):
                all_sources.append({"type": s.get("type") or "live","title": s.get("title") or "Source","url": s.get("url")})
        for it in image_items:
            all_sources.append({"type":"image","title": it.get("title") or "Image","url": it.get("url")})

        seen, dedup = set(), []
        for s in all_sources:
            key = (s.get("type"), s.get("title"), s.get("url"))
            if not s.get("url") or key in seen: continue
            seen.add(key); dedup.append(s)

        lines = []
        if kb_take:
            lines.append("Knowledge base context:"); lines += [f"- {x}" for x in kb_take]; lines.append("")
        if live_take:
            lines.append("Retrieved context (live search):"); lines += [f"- {x}" for x in live_take]; lines.append("")
        if api_take:
            lines.append("Multi-API context:"); lines += [f"- {x}" for x in api_take]; lines.append("")
        if img_take:
            lines.append("Image context (captions):"); lines += [f"- {x}" for x in img_take]; lines.append("")
        if user_take:
            lines.append("User images (text-only summary):"); lines += [f"{x}" for x in user_take]; lines.append("")

        numbered, n = [], 1
        if dedup:
            lines.append("Sources (numbered):")
            for s in dedup:
                title = s.get("title") or s.get("url") or "Source"
                url = s.get("url") or ""
                lines.append(f"[{n}] {title} — {url}")
                numbered.append({"n": n, **s}); n += 1

        context_block = "\n".join(lines).strip()
        if len(context_block) > max_context_chars:
            context_block = context_block[:max_context_chars].rsplit("\n", 1)[0].strip()
        return context_block, numbered
    def _build_messages(self, system_prompt: str, user_prompt: str, context_block: str,
                        thinking: bool, multimodal: bool, images_b64_list: Optional[list],
                        answer_lang: Optional[str], context_messages_json: str):
        msgs = []
        sys = (system_prompt or "").strip()
        if answer_lang:
            sys = (sys + f"\nAnswer strictly in '{answer_lang}' unless explicitly asked otherwise.").strip()
        if sys:
            msgs.append({"role":"system","content": sys})

        cm = (context_messages_json or "").strip()
        if cm:
            try:
                arr = json.loads(cm)
                if isinstance(arr, list):
                    for m in arr:
                        if isinstance(m, dict) and "role" in m and "content" in m:
                            msgs.append({"role": m["role"], "content": str(m["content"])})
            except Exception as e:
                msgs.append({"role":"system","content": f"Note: previous context parse failed: {e}"})

        if context_block:
            msgs.append({"role":"system","content": "Use the following context and cite [n] where applicable.\n\n" + context_block})

        user_content = user_prompt or ""
        if thinking:
            user_content = "Provide your chain-of-thought between <think>...</think> and the final answer between <final>...</final>.\n\n" + user_content

        if multimodal and images_b64_list:
            msgs.append({"role":"user","content": user_content, "images": images_b64_list})
        else:
            msgs.append({"role":"user","content": user_content})
        return msgs

    def _extract_thinking_and_result(self, data: Any, thinking_enabled: bool = False):
        text, think_out = "", ""
        if isinstance(data, dict):
            if thinking_enabled and data.get("thinking"):
                think_out = str(data.get("thinking")).strip()
            for key in ["response", "display_text", "output"]:
                if data.get(key):
                    text = str(data[key]); break
            if not text and "message" in data:
                msg = data["message"]
                if isinstance(msg, dict): text = msg.get("content","") or ""
                elif isinstance(msg, str): text = msg
            if not text and isinstance(data.get("choices"), list) and data["choices"]:
                ch = data["choices"][0]
                if isinstance(ch, dict):
                    if isinstance(ch.get("message"), dict):
                        text = ch["message"].get("content", "")
                    elif "text" in ch:
                        text = ch["text"]
        if not text and isinstance(data, str): text = data
        if not text and isinstance(data, dict):
            try: text = json.dumps(data, ensure_ascii=False)
            except Exception: text = ""
        text = (text or "").strip()
        final_out = text

        if thinking_enabled:
            m1 = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL|re.IGNORECASE)
            m2 = re.search(r"<final>(.*?)</final>", text, flags=re.DOTALL|re.IGNORECASE)
            if m1:
                think_tag_content = m1.group(1).strip()
                think_out = (think_out + "\n" + think_tag_content).strip() if think_out else think_tag_content
            if m2:
                final_out = m2.group(1).strip()
            elif m1:
                final_out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL|re.IGNORECASE).strip()
            if not think_out and text:
                parts = re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z])", text, maxsplit=1)
                if len(parts) == 2:
                    think_out, final_out = parts[0].strip(), parts[1].strip()
                else:
                    think_out = "[Model did not expose a separate chain-of-thought]"
        else:
            think_out = ""
        return think_out, final_out

    def _sanitize_model_output(self, s: str) -> str:
        if not s: return s
        rx_sentence = re.compile(r'^\s*Sentence\s+\d+\s*[:=：]', re.IGNORECASE)
        out_lines = []
        for raw in s.splitlines():
            line = raw.rstrip("\r")
            stripped = line.strip()
            if stripped.startswith("[DEBUG]"): continue
            if stripped.lower().startswith("total_duration:"): continue
            if rx_sentence.match(stripped): continue
            if "Aristomenis Marinis presents" in stripped: continue
            out_lines.append(line)
        normalized, blank = [], False
        for ln in out_lines:
            if ln.strip() == "":
                if blank: continue
                blank = True; normalized.append("")
            else:
                blank = False; normalized.append(ln)
        out = "\n".join(normalized)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out
    def _image_search(self, provider, query, max_results, timeout, keys):
        provider = (provider or "off").lower()
        if provider == "off" or not query: return []
        try:
            if provider == "duckduckgo":
                headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://duckduckgo.com/"}
                s = requests.Session()
                s.get("https://duckduckgo.com/", headers=headers, timeout=5)
                r = s.get("https://duckduckgo.com/i.js",
                          params={"q": query, "o":"json"}, headers=headers, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                out = []
                for it in data.get("results", [])[:max_results]:
                    out.append({
                        "title": it.get("title") or "Image",
                        "image": it.get("image"),
                        "thumbnail": it.get("thumbnail"),
                        "url": it.get("url") or it.get("source")
                    })
                return out
            if provider == "unsplash":
                key = (keys or {}).get("unsplash") or ""
                if not key: return []
                r = requests.get("https://api.unsplash.com/search/photos",
                                 params={"query": query, "per_page": max_results},
                                 headers={"Authorization": f"Client-ID {key}","Accept-Version":"v1"},
                                 timeout=timeout)
                r.raise_for_status()
                data = r.json()
                out = []
                for it in data.get("results", [])[:max_results]:
                    out.append({
                        "title": it.get("description") or it.get("alt_description") or "Unsplash image",
                        "image": it.get("urls",{}).get("regular"),
                        "thumbnail": it.get("urls",{}).get("thumb"),
                        "url": it.get("links",{}).get("html")
                    })
                return out
            if provider == "bing":
                key = (keys or {}).get("bing") or ""
                if not key: return []
                r = requests.get("https://api.bing.microsoft.com/v7.0/images/search",
                                 params={"q": query, "count": max_results},
                                 headers={"Ocp-Apim-Subscription-Key": key}, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                out = []
                for it in data.get("value", [])[:max_results]:
                    out.append({
                        "title": it.get("name") or "Bing image",
                        "image": it.get("contentUrl"),
                        "thumbnail": it.get("thumbnailUrl"),
                        "url": it.get("hostPageUrl")
                    })
                return out
            if provider == "pexels":
                key = (keys or {}).get("pexels") or ""
                if not key: return []
                r = requests.get("https://api.pexels.com/v1/search",
                                 params={"query": query, "per_page": max_results},
                                 headers={"Authorization": key}, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                out = []
                for it in data.get("photos", [])[:max_results]:
                    out.append({
                        "title": it.get("alt") or "Pexels image",
                        "image": it.get("src",{}).get("large"),
                        "thumbnail": it.get("src",{}).get("tiny"),
                        "url": it.get("url")
                    })
                return out
        except Exception as e:
            self._logger.debug(f"[AIO] Image search error ({provider}): {e}")
        return []

    def _process_optional_image_input(self, optional_image_input, is_multimodal, images_b64_list, user_images_info):
        if optional_image_input is None: return
        try:
            try:
                import torch
                if isinstance(optional_image_input, torch.Tensor):
                    arr = optional_image_input.detach().cpu().numpy()
                else:
                    arr = optional_image_input
            except Exception:
                arr = optional_image_input
            if not isinstance(arr, np.ndarray): return
            if arr.ndim == 3: arr = arr[None, ...]
            for i, img in enumerate(arr):
                b64 = _png_b64_from_ndarray(img)
                if is_multimodal:
                    images_b64_list.append(b64)
                    user_images_info.append(f"optional_image_input[{i+1}]")
        except Exception as e:
            self._logger.debug(f"[AIO] Error processing IMAGE: {e}")
    def _is_empty_oss(self, data):
        if not isinstance(data, dict): return False
        resp = (data.get("response") or "").strip()
        done_reason = (data.get("done_reason") or "").lower()
        return resp == "" and (done_reason in ("load","") or data.get("done") in (True, False))

    def _retry_simple_generate(self, base, model, messages, options, keep, timeout_s):
        system = ""; user = ""
        for m in messages:
            if m.get("role") == "system": system += (m.get("content") or "") + "\n\n"
            elif m.get("role") == "user": user += (m.get("content") or "") + "\n\n"
        payload = {"model": model, "system": system.strip(), "prompt": user.strip(),
                   "options": {k: v for k, v in options.items() if k not in ("format","raw")},
                   "stream": False, "keep_alive": keep}
        try:
            r = requests.post(f"{base}/api/generate", json=payload, headers=_ua(), timeout=timeout_s)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self._logger.debug(f"[AIO] simple generate retry failed: {e}")
            return {"response": ""}

    def _ollama_chat(self, base_url: str, model: str, messages: list, options: dict,
                     keep_alive: Any, keep_alive_unit: str, timeout_s: int = 120):
        base = (base_url or "http://127.0.0.1:11434").rstrip("/")
        if isinstance(keep_alive, int):
            keep = "24h" if keep_alive < 0 else f"{keep_alive}{'h' if keep_alive_unit == 'hours' else 'm'}"
        else:
            keep = "10m"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep,
            "options": options or {}
        }
        try:
            r = requests.post(f"{base}/api/chat", json=payload, headers=_ua(), timeout=timeout_s)
            r.raise_for_status()
            res = r.json()
            if self._is_empty_oss(res):
                time.sleep(0.6)
                return self._retry_simple_generate(base, model, messages, options, keep, timeout_s)
            return res
        except Exception as e:
            try:
                system = ""; user = ""
                for m in messages:
                    if m.get("role") == "system": system += (m.get("content") or "") + "\n\n"
                    elif m.get("role") == "user": user += (m.get("content") or "") + "\n\n"
                gen_payload = {"model": model, "system": system.strip(), "prompt": user.strip(),
                               "options": options or {}, "stream": False, "keep_alive": keep}
                r2 = requests.post(f"{base}/api/generate", json=gen_payload, headers=_ua(), timeout=timeout_s)
                r2.raise_for_status()
                return r2.json()
            except Exception as e2:
                return {"response": f"[AIO] Failed to retrieve response (fallback): {e2}"}
    def _collect_all_context(self, user_prompt: str, use_live_search: bool,
                             wiki_lang: str, search_max_results: int, search_timeout_s: int,
                             enhanced_live_fallbacks: bool, brave_key: str, serper_key: str,
                             google_cse_key: str, google_cse_cx: str,
                             image_provider: str, image_max_results: int, unsplash_key: str, bing_key: str, pexels_key: str,
                             use_weather_apis: bool, use_nature_apis: bool, use_finance_apis: bool, use_news_apis: bool, use_media_apis: bool,
                             use_general_apis: bool, use_scraping_fallback: bool,
                             newsapi_key: str, guardian_key: str, nytimes_key: str):
        live_snips, live_sources, api_snips, image_items = [], [], [], []

        if use_live_search:
            if enhanced_live_fallbacks:
                s, src = self._collect_live_enhanced(
                    user_prompt, max_results=search_max_results, timeout=search_timeout_s, wiki_lang=wiki_lang,
                    brave_key=brave_key, serper_key=serper_key, google_cse_key=google_cse_key, google_cse_cx=google_cse_cx
                )
            else:
                s, src = self._collect_live_once(
                    user_prompt, max_results=search_max_results, timeout=search_timeout_s, wiki_lang=wiki_lang
                )
            live_snips.extend(s); live_sources.extend(src)

        # Core free router (existing)
        if use_weather_apis:
            w = self._weather_with_fallbacks(user_prompt, timeout=search_timeout_s)
            if w: api_snips.append(w)

        # Optional: add your previous category routers here (nature/finance/news/media/general)

        # Key-based news/web enrichment (optional)
        for fn, key, lab in [
            (self.news_api_org if hasattr(self, "news_api_org") else None, newsapi_key, "newsapi"),
            (self.guardian_api if hasattr(self, "guardian_api") else None, guardian_key, "guardian"),
            (self.nytimes_api if hasattr(self, "nytimes_api") else None, nytimes_key, "nytimes"),
        ]:
            if fn and key:
                s, c = fn(user_prompt, key, max_results=min(5, search_max_results), timeout=min(12, search_timeout_s+2))
                if s: api_snips.append(s); live_sources += c

        # Image search
        if image_provider and image_provider != "off":
            image_items = self._image_search(
                image_provider, user_prompt, image_max_results, search_timeout_s,
                keys={"unsplash": unsplash_key, "bing": bing_key, "pexels": pexels_key},
            )
            if self._safe_mode:
                image_items = [im for im in (image_items or []) if self._safe_allow(im.get("title",""), im.get("image") or im.get("url") or "")]

        return live_snips, live_sources, api_snips, image_items

    def run(self, url, model_choice, model_override, list_models_on_run,
            refresh_connection, keep_alive, keep_alive_unit,
            async_enabled, cache_ttl_s, profile, log_to_file, log_file_path,
            use_live_search, wiki_lang, search_query_override, search_max_results, search_timeout_s, enhanced_live_fallbacks,
            brave_key, serper_key, google_cse_key, google_cse_cx,
            use_weather_apis, use_nature_apis, use_finance_apis, use_news_apis, use_media_apis, use_general_apis, use_scraping_fallback,
            use_animals_apis, use_anime_apis, use_books_apis, use_crypto_apis, use_fx_apis, use_devfun_apis, use_geoip_apis, use_music_apis, use_sports_apis,
            per_category_limit, safe_content_mode, allow_nsfw_registry,
            newsapi_key, guardian_key, nytimes_key,
            use_knowledge_base, knowledge_base_dir, kb_max_chunks, kb_chunk_chars, kb_overlap_chars,
            image_search_provider, image_max_results, unsplash_key, bing_key, pexels_key,
            use_tool_prompt, thinking, gpt_oss_harmonizer, legacy_weather_mode,
            translate_before_api, force_output_english,
            system_prompt, user_prompt,
            temperature, num_predict, mirostat, top_k, top_p, repeat_penalty, max_context_chars,
            context_messages_json, api_timeout_override_s, detect_multimodal, debug,
            optional_image_input=None, images_b64_json="", optional_image_b64="", batch_image_paths="[]"):

        self._safe_mode = bool(safe_content_mode)
        self._ttl_default = int(cache_ttl_s or 180)
        self._logger = _setup_logger(bool(debug or log_to_file), path=(log_file_path if log_to_file else ""))

        base_url = (url or "http://127.0.0.1:11434").strip().rstrip("/")
        if refresh_connection:
            self._health_check(base_url)
        if list_models_on_run and debug:
            self._logger.debug(f"[AIO] Models: {self._list_ollama_models(base_url, timeout=6)}")

        model = (model_override or model_choice or "").strip() or "llama3.1"
        prof = self._model_profile(model)
        ctx_limit = int(max(1200, min(700000, max_context_chars)))

        # Profiles
        if profile == "speed":
            search_timeout_s = max(6, search_timeout_s - 3)
            per_category_limit = max(1, min(2, per_category_limit))
        elif profile == "max_context":
            ctx_limit = min(200000, max(ctx_limit, 20000))
        elif profile == "low_latency":
            search_timeout_s = max(5, min(8, search_timeout_s))
            async_enabled = True
            per_category_limit = 2

        # Prompt injection tuned for mistral
        local_thinking = bool(thinking and not prof["is_mistral"])
        system_prompt = self._inject_toolprompt(system_prompt, bool(use_tool_prompt and not prof["is_mistral"]), local_thinking)

        # Use original prompt for search/weather (avoid translation artifacts)
        orig_user_prompt = user_prompt
        effective_query = (search_query_override.strip() or orig_user_prompt)

        # KB
        kb_hits = []
        if use_knowledge_base and knowledge_base_dir.strip():
            try:
                self._ensure_kb(knowledge_base_dir.strip(), kb_chunk_chars, kb_overlap_chars)
                # Use EN or original? Use EN for KB matching, but we translate after collecting contexts
                kb_query_en, _, _ = self._translate_if_needed(orig_user_prompt, target="en", enable=True)
                kb_hits = self._kb_search(kb_query_en, k=kb_max_chunks)
            except Exception as e:
                self._logger.debug(f"[AIO] KB error: {e}")

        # Live/API/Image with legacy weather toggle behavior
        self._legacy_weather_mode_runtime = bool(legacy_weather_mode)
        live_snips, live_sources, api_snips, image_items = self._collect_all_context(
            user_prompt=effective_query, use_live_search=use_live_search,
            wiki_lang=wiki_lang, search_max_results=search_max_results, search_timeout_s=search_timeout_s,
            enhanced_live_fallbacks=enhanced_live_fallbacks,
            brave_key=brave_key, serper_key=serper_key, google_cse_key=google_cse_key, google_cse_cx=google_cse_cx,
            image_provider=image_search_provider, image_max_results=image_max_results,
            unsplash_key=unsplash_key, bing_key=bing_key, pexels_key=pexels_key,
            use_weather_apis=use_weather_apis, use_nature_apis=use_nature_apis, use_finance_apis=use_finance_apis,
            use_news_apis=use_news_apis, use_media_apis=use_media_apis, use_general_apis=use_general_apis,
            use_scraping_fallback=use_scraping_fallback,
            newsapi_key=newsapi_key, guardian_key=guardian_key, nytimes_key=nytimes_key
        )

        # Registry bulk (tens of APIs) — uses original prompt and SFW guard
        reg_snips, reg_sources = self._collect_registry_bulk(
            query=effective_query,
            cfg={
                "use_animals_apis": use_animals_apis,
                "use_anime_apis": use_anime_apis,
                "use_books_apis": use_books_apis,
                "use_crypto_apis": use_crypto_apis,
                "use_fx_apis": use_fx_apis,
                "use_devfun_apis": use_devfun_apis,
                "use_geoip_apis": use_geoip_apis,
                "use_music_apis": use_music_apis,
                "use_sports_apis": use_sports_apis,
                "per_category_limit": per_category_limit,
                "search_timeout_s": search_timeout_s,
            }
        )

        api_snips = (api_snips or []) + reg_snips
        live_sources = (live_sources or []) + reg_sources

        # Multimodal images in
        images_b64_list, user_images_info = [], []
        try:
            if images_b64_json and images_b64_json.strip():
                arr = json.loads(images_b64_json)
                if isinstance(arr, list):
                    for it in arr:
                        if isinstance(it, str) and len(it) > 24:
                            images_b64_list.append(it); user_images_info.append("json image")
        except Exception as e:
            self._logger.debug(f"[AIO] images_b64_json parse error: {e}")

        if optional_image_b64 and optional_image_b64.strip():
            images_b64_list.append(optional_image_b64.strip())
            user_images_info.append("optional image b64")

        try:
            arr = json.loads(batch_image_paths or "[]")
            if isinstance(arr, list):
                for path in arr:
                    if isinstance(path, str) and os.path.isfile(path):
                        with Image.open(path) as im:
                            im = im.convert("RGB")
                            buf = io.BytesIO(); im.save(buf, format="PNG")
                            images_b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
                            user_images_info.append(f"batch image: {os.path.basename(path)}")
        except Exception as e:
            self._logger.debug(f"[AIO] batch_image_paths parse error: {e}")

        is_mm = self._is_probably_multimodal(base_url, model, allow_probe=True) if detect_multimodal else False
        self._process_optional_image_input(optional_image_input, is_mm, images_b64_list, user_images_info)

        # Build context
        context_block, numbered_sources = self._build_context_v3(
            kb_hits=kb_hits,
            live_snips=live_snips,
            api_snips=api_snips,
            live_sources=live_sources,
            image_items=[],  # images only as sources; keep text minimal
            user_images_info=user_images_info,
            max_context_chars=ctx_limit
        )

        if use_live_search and not live_snips and not live_sources:
            context_block = ("Note: no live context found. Be brief and avoid speculation.\n\n" + (context_block or "")).strip()

        # Translate after context acquisition (prompt only)
        user_prompt_en, lang_detected, was_translated = self._translate_if_needed(
            orig_user_prompt, target="en", enable=translate_before_api
        )

        # Messages (force EN)
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt_en,
            context_block=context_block,
            thinking=local_thinking,
            multimodal=is_mm,
            images_b64_list=images_b64_list,
            answer_lang="en" if self._safe_bool(force_output_english, True) else "en",
            context_messages_json=context_messages_json
        )
        messages = self._harmonize_for_gptoss(model, messages, enabled=gpt_oss_harmonizer)

        # Ollama opts + timeout
        ollama_opts = {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
            "mirostat": int(mirostat),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repeat_penalty": float(repeat_penalty),
        }
        chat_timeout = max(30, min(240, search_timeout_s * 12))
        if isinstance(api_timeout_override_s, int) and api_timeout_override_s > 0:
            chat_timeout = api_timeout_override_s

        data = self._ollama_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            options=ollama_opts,
            keep_alive=int(keep_alive),
            keep_alive_unit=keep_alive_unit,
            timeout_s=chat_timeout
        )

        thinking_out, final_out = self._extract_thinking_and_result(data, thinking_enabled=local_thinking)
        thinking_out = self._sanitize_model_output(thinking_out)
        final_out = self._sanitize_model_output(final_out)

        if not final_out.strip():
            final_out = "No content returned by the model. Try refreshing connection or switching model. If using OSS, wait a moment and retry."

        sources_json = json.dumps(numbered_sources, indent=2, ensure_ascii=False)
        return thinking_out, final_out, sources_json
NODE_CLASS_MAPPINGS = {
    "OllamaSuperAIO": OllamaSuperAIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaSuperAIO": "Ollama Super AIO (Live+MultiAPI+KB+Images+Translate+Thinking+Async)",
}
