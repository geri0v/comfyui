import os
import re
import io
import json
import time
import base64
import hashlib
import random
import logging
import glob
import requests
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from functools import lru_cache

def ua():
    return 'User-Agent: ComfyUI-Ollama-SuperAIO7.0'

def clamps(s, limit):
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    if limit <= 0:
        return s
    if len(s) <= limit:
        return s
    return s[:limit].rsplit(' ', 10)[0]

def now():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

def setuplogger(enabled: bool, path: str = ""):
    logger = logging.getLogger("SuperAIO")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)
    h = logging.StreamHandler()
    fmt = logging.Formatter('%(levelname)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    if enabled and path:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

class OllamaSuperAIO:
    CATEGORY = "AIOllama"
    RETURNTYPES = ["text", "json"]
    RETURNNAMES = ["answer", "sources"]

    def __init__(self, config: Optional[dict] = None):
        self.kbchunks: List[dict] = []
        self.kbidf: Dict[str, float] = {}
        self.kbready: bool = False
        self.kbcachesig: str = ""
        self.ttlcache: Dict[str, Tuple[float, Any, int]] = {}
        self.ttldefault: int = 180
        self.safemode: bool = True
        self.logger = setuplogger(True)
        self.runid: str = ""
        self.ctxlimit = 1800
        self.temperature = 0.7
        self.topp = 0.9
        self.topk = 40
        self.numpredict = 512
        self.searchtimeouts = 10
        self.percategorylimit = 3
        self.asyncenabled = False

    def runidnew(self):
        self.runid = f"AIO-{int(time.time()*1000)}"
        self.logger.info(f"runid: {self.runid}")

    def modelprofile(self, name: str):
        n = (name or "").lower()
        return {
            "ismistral": "mistral" in n,
            "isgptoss": "gpt-oss" in n or "oss" in n,
            "isdeepseek": "deepseek" in n,
            "isqwen": "qwen" in n,
        }

    @classmethod
    def bootstrapmodelsfordropdown(cls, base="http://127.0.0.1:11434"):
        if hasattr(cls, "cachedmodelchoices") and hasattr(cls, "cachedmodelfetchts"):
            if getattr(cls, "cachedmodelchoices") and time.time() - getattr(cls, "cachedmodelfetchts") < 120:
                return cls.cachedmodelchoices
        try:
            data = requests.get(f"{base.rstrip('/')}/api/tags", headers={"User-Agent": ua()}, timeout=2).json()
            models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
            if not models:
                models = ["llama3.1", "mistral", "qwen2.5-vl", "deepseek-r17b", "gpt-oss", "llava"]
        except Exception:
            models = ["llama3.1", "mistral", "qwen2.5-vl", "deepseek-r17b", "gpt-oss", "llava"]
        cls.cachedmodelchoices = models
        cls.cachedmodelfetchts = time.time()
        return models

    def healthcheck(self, baseurl: str, timeout=6):
        try:
            r = requests.get(f"{baseurl.rstrip('/')}/api/tags", headers={"User-Agent": ua()}, timeout=timeout)
            r.raise_for_status()
            return True
        except Exception as e:
            self.logger.debug(f"AIO Health check failed: {e}")
            return False

    def withretry(self, func, tries=2, delay=0.35, backoff=1.6, jitter=0.1, *args, **kwargs):
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

    def cacheget(self, key):
        rec = self.ttlcache.get(key)
        if not rec:
            return None
        ts, val, ttl = rec
        if time.time() - ts > ttl:
            self.ttlcache.pop(key, None)
            return None
        return val

    def cacheput(self, key, value, ttl=None):
        self.ttlcache[key] = (time.time(), value, ttl or self.ttldefault)

    def cachekey(self, name, params):
        try:
            return name + hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        except Exception:
            return name + str(params)

    def detectlanguage(self, text: str):
        try:
            r = requests.post("https://libretranslate.com/detect", data={"q": text}, timeout=6).json()
            if isinstance(r, list) and len(r) > 0:
                return r[0].get("language", "en")
        except Exception:
            pass
        return "en"

    def translatedeepl(self, text: str, target="EN", apikey=""):
        try:
            if not apikey or not text:
                return None
            r = requests.post(
                "https://api-free.deepl.com/v2/translate",
                data={"auth_key": apikey, "text": text, "target_lang": target.upper()},
                timeout=8,
                headers={"User-Agent": ua()},
            )
            r.raise_for_status()
            data = r.json()
            if "translations" in data and len(data["translations"]) > 0:
                return data["translations"][0]["text"]
        except Exception:
            return None

    def translatelibres(self, text: str, target="EN"):
        try:
            r = requests.post(
                "https://libretranslate.com/translate",
                data={"q": text, "source": "auto", "target": target, "format": "text"},
                timeout=8,
                headers={"User-Agent": ua(), "Accept": "application/json"},
            )
            r.raise_for_status()
            return r.json().get("translatedText", None)
        except Exception:
            return None

    def translateifneeded(self, text: str, target="EN", enable=True):
        if not enable or not text:
            return text
        try:
            lang_detected = self.detectlanguage(text)
            if lang_detected == target:
                return text
            deepl_api_key = os.environ.get("DEEPL_API_KEY", "").strip()
            if deepl_api_key:
                translated = self.translatedeepl(text, target, deepl_api_key)
                if translated:
                    return translated
            translated = self.translatelibres(text, target)
            if translated:
                return translated
        except Exception as e:
            self.logger.debug(f"Translation error: {e}")
        return text

    def buildkbindex(self, kbdir: str, chunkchars=900, overlapchars=120):
        files = sorted(glob.glob(os.path.join(kbdir, "**/*.txt"), recursive=True) + 
                       glob.glob(os.path.join(kbdir, "**/*.md"), recursive=True))
        items = []
        for path in files:
            try:
                st = os.stat(path)
                items.append(f"{path}{int(st.st_mtime)}{st.st_size}")
            except Exception:
                continue
        sigstr = "".join(items) + f"{chunkchars}{overlapchars}"
        sig = hashlib.sha256(sigstr.encode("utf-8")).hexdigest()
        if sig == self.kbcachesig and self.kbready:
            return
        chunks = []
        for path in files:
            text = self.readtextfiles(path)
            if not text:
                continue
            chunks += self.chunktext(text, chunkchars, overlapchars)
        self.kbchunks = chunks
        self.kbready = True
        self.kbcachesig = sig

    def readtextfiles(self, path: str) -> Optional[str]:
        try:
            with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return None

    def chunktext(self, text: str, chunksize=900, overlap=120) -> list:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + chunksize, n)
            seg = text[i:end]
            m = re.search(r"[.!?]", seg[::-1])
            if m:
                pos = end - m.start()
                if pos - i > chunksize * 0.6:
                    end = pos
            chunk = text[i:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            i = max(end - overlap, i + 1)
        return chunks

    def tokenizes(self, s: str) -> list:
        # Simple tokenizer, lowercases and splits on alphanumerics
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", s)]

    def ranktfidf(self, query: str, chunks: List[str], kbidf: Dict[str, float]):
        scored = []
        qtokens = self.tokenizes(query)
        qtf = Counter(qtokens)
        normqtf = {k: v / len(qtokens) for k, v in qtf.items()}
        for chunk in chunks:
            ctokens = self.tokenizes(chunk)
            tf = Counter(ctokens)
            denom = len(ctokens)
            tfidf_score = 0.0
            for term in qtokens:
                tf_term = tf.get(term, 0) / denom if denom > 0 else 0
                idf_term = kbidf.get(term, 0)
                tfidf_score += normqtf.get(term, 0) * tf_term * idf_term
            scored.append((chunk, tfidf_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, score) for chunk, score in scored[:10]]

    def estimatecontextbudgets(self, maxcontextchars):
        if maxcontextchars <= 4000:
            return {'kb': int(maxcontextchars * 0.40), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.18), 'images': int(maxcontextchars * 0.06),
                    'userimg': int(maxcontextchars * 0.06)}
        elif maxcontextchars <= 160000:
            return {'kb': int(maxcontextchars * 0.36), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.20), 'images': int(maxcontextchars * 0.07),
                    'userimg': int(maxcontextchars * 0.07)}
        else:
            return {'kb': int(maxcontextchars * 0.32), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.24), 'images': int(maxcontextchars * 0.07),
                    'userimg': int(maxcontextchars * 0.07)}

    def buildcontextv3(self, kbhits, livesnips, apisnips, livesources, imageitems, userimagesinfo, maxcontextchars=3600):
        budgets = self.estimatecontextbudgets(maxcontextchars)
        apitakeweather = 650
        apitakeother = budgets.get('api', 2000) - apitakeweather
        context_lines = []

        if len(apisnips) > 0:
            apitake = []
            weatherlines = [s for s in apisnips if s.startswith('Weather')]
            otherlines = [s for s in apisnips if not s.startswith('Weather')]

            apitake += weatherlines[:apitakeweather]
            apitake += otherlines[:apitakeother]

            context_lines.extend(apitake)

        context_lines.extend(kbhits[:budgets.get('kb', 900)])
        context_lines.extend(livesnips[:budgets.get('live', 1000)])
        context_lines.extend(livesources[:budgets.get('live', 1000)])
        context_lines.extend(imageitems)
        context_lines.extend(userimagesinfo)

        joined_context = "\n".join(context_lines)
        if len(joined_context) > maxcontextchars:
            joined_context = joined_context[:maxcontextchars]
            joined_context = joined_context.rsplit(' ', 1)[0]
        return joined_context

    def listollamamodels(self, baseurl, timeout=6):
        try:
            data = requests.get(f"{baseurl.rstrip('/')}/api/tags", headers={"User-Agent": ua()}, timeout=timeout).json()
            return [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
        except Exception as e:
            self.logger.debug(f"Failed to list models: {e}")
            return []

    def create_registry_categories(self):
        return [
            "animals", "books", "crypto", "devfun", "finance", "fx",
            "general", "geoip", "media", "music", "nature", "news",
            "sports", "anime"
        ]

    def runregistrycategory(self, category, query, limit=3, timeout=8):
        items = []
        try:
            if category == "animals":
                url = f"https://catfact.ninja/fact"
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                fact = r.json().get("fact", "")
                items.append(fact)
            # Implement other categories similarly as needed
        except Exception as e:
            self.logger.debug(f"Registry API error for {category}: {e}")
        return items[:limit]

    def listregistrycategories(self):
        return self.create_registry_categories()

    def collectliveonce(self, query: str, maxresults=5, timeout=10, wikilang="en", ddg=None):
        snippets = []
        sources = []
        if not query.strip():
            return snippets, sources
        if ddg is None:
            try:
                import duckduckgo_search
                ddg = duckduckgo_search.DuckDuckGoSearch()
            except ImportError:
                self.logger.debug("duckduckgo_search library not installed")
                return snippets, sources
        try:
            abstxt = ddg.get_instant(query, timeout=timeout)
            absurl = ddg.get_instant_url(query, timeout=timeout)
            if abstxt:
                snippets.append(abstxt.strip())
            if absurl:
                sources.append(("duckduckgo", "Instant Answer", absurl))
            related = ddg.get_related(query, maxresults=maxresults, timeout=timeout)
            for rel in related:
                if isinstance(rel, dict) and "text" in rel and "url" in rel:
                    snippets.append(rel["text"])
                    sources.append(("duckduckgo", "Related", rel["url"]))
        except Exception as e:
            self.logger.debug(f"DuckDuckGo search failed: {e}")

        try:
            from wikipediaapi import Wikipedia
            wiki = Wikipedia(language=wikilang)
            page = wiki.page(query)
            if page.exists():
                summary = page.summary[0:1000]
                snippets.append(summary)
                sources.append(("wikipedia", "Page", page.fullurl))
        except Exception as e:
            self.logger.debug(f"Wikipedia search failed: {e}")

        return snippets, sources

    def sanitizemodeloutput(self, s: str):
        if not s:
            return s
        outlines = []
        rxsentence = re.compile(r'^[A-Z][a-z].*[\.\?\!]$')
        for raw in s.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("DEBUG") or stripped.lower().startswith("totalduration"):
                continue
            if rxsentence.match(stripped):
                continue
            if "Aristomenis Marinis presents" in stripped:
                continue
            outlines.append(line)
        normalized = "\n".join(outlines)
        normalized = re.sub(r"\n\s*\n", "\n\n", normalized)
        return normalized.strip()

    def reformat_thinking(self, text: str):
        thinking_pattern = re.compile(r"(?si)^think[^\w]*(.*)final[^\w]*(.*)", re.DOTALL | re.IGNORECASE)
        m = thinking_pattern.match(text)
        if m:
            think_output = m.group(1).strip()
            final_output = m.group(2).strip()
            return think_output, final_output
        return text, None

    def harmonizeforgptossmodel(self, messages, enabled=True):
        if not enabled:
            return messages
        new_msgs = []
        for m in messages:
            if m.get("role") == "assistant":
                m["role"] = "system"
            new_msgs.append(m)
        return new_msgs

    def buildmessagessystemuser(self, systemprompt: str, userprompt: str, contextblock: str,
                                thinking: bool, multimodal: bool, imagesb64list: list,
                                answerlang: Optional[str] = None, contextmessagesjson: str = None):
        sys_prompt = (systemprompt or "").strip()
        if answerlang:
            sys_prompt += f"\nAnswer language: {answerlang}"
        messages = [{"role": "system", "content": sys_prompt}]
        user_content = userprompt.strip() + "\n\nContext:\n" + contextblock
        if contextmessagesjson:
            try:
                extra_messages = json.loads(contextmessagesjson)
                if isinstance(extra_messages, list):
                    messages.extend(extra_messages)
            except Exception:
                pass
        messages.append({"role": "user", "content": user_content})
        if thinking:
            messages.append({"role": "user", "content": "[Thinking enabled]"})
        if multimodal and imagesb64list:
            for img_b64 in imagesb64list:
                messages.append({"role": "user", "content": {"image": img_b64}})
        return messages

    def generate(self, baseurl, model, messages, options=None, stream=False, keepalive=0):
        url = f"{baseurl.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keepalive": keepalive,
            **(options or {})
        }
        try:
            response = requests.post(url, json=payload, headers={"User-Agent": ua()}, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Generation API call failed: {e}")
            return None

    def harmonize_response(self, response):
        if not response:
            return None
        if isinstance(response, dict) and "choices" in response:
            text = response["choices"][0].get("message", {}).get("content", "")
            return text
        return response

    def run(self, *args, **kwargs):
        self.logger.info("Starting run()")
        baseurl = kwargs.get("baseurl", "http://127.0.0.1:11434")
        model = kwargs.get("model", "llama3.1")
        userprompt = kwargs.get("userprompt", "")
        systemprompt = kwargs.get("systemprompt", "")
        multimodal = kwargs.get("multimodal", False)
        optionalimageinput = kwargs.get("optionalimageinput", None)
        maxcontextchars = kwargs.get("maxcontextchars", 3600)
        timeout = kwargs.get("timeout", 60)

        # Prepare images
        imagesb64list = []
        userimagesinfo = []
        ismultimodal = self.isprobablymultimodal(baseurl, model)
        self.processoptionalimageinput(optionalimageinput, ismultimodal, imagesb64list, userimagesinfo)

        # Collect context (placeholder: no kb or live search implemented in this snippet)
        contextblock, weather = self.collectallcontextparallel(
            effectivequery=userprompt,
            cfg=kwargs,
            livesnips=[],
            livesources=[],
            apisnips=[],
            imageitems=imagesb64list,
            maxcontextchars=maxcontextchars,
        )

        messages = self.buildmessagessystemuser(
            systemprompt=systemprompt,
            userprompt=userprompt,
            contextblock=contextblock,
            thinking=kwargs.get("thinking", False),
            multimodal=multimodal,
            imagesb64list=imagesb64list,
            answerlang=kwargs.get("answerlang", None),
            contextmessagesjson=kwargs.get("contextmessagesjson", None),
        )

        response = self.generate(baseurl, model, messages, stream=kwargs.get("stream", False), keepalive=kwargs.get("keepalive", 0))
        output = self.harmonize_response(response)
        finaloutput = self.sanitizemodeloutput(output)

        return finaloutput, json.dumps({"weather": weather}, indent=2)

    def listollamamodels(self, baseurl, timeout=6):
        try:
            data = requests.get(f"{baseurl.rstrip('/')}/api/tags", headers={"User-Agent": ua()}, timeout=timeout).json()
            return [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
        except Exception as e:
            self.logger.debug(f"Failed to list models: {e}")
            return []

    def isprobablymultimodal(self, baseurl, model, allowprobe=True):
        if not model:
            return False
        multimodal_suffixes = ["llava", "qwen-vl", "blip", "paligemma", "moondream", "llama-vision", "llama3.2-vision"]
        model_lower = model.lower()
        if any(x in model_lower for x in multimodal_suffixes):
            return True
        if not allowprobe:
            return False
        try:
            data = requests.get(f"{baseurl.rstrip('/')}/api/tags", headers={"User-Agent": ua()}, timeout=3).json()
            models = data.get("models", [])
            for m in models:
                if m.get("name", "").lower() == model_lower:
                    return "vision" in m.get("tags", [])
        except Exception:
            pass
        return False

    def processoptionalimageinput(self, optionalimageinput, ismultimodal, imagesb64list, userimagesinfo):
        if optionalimageinput is None:
            return
        try:
            import PIL.Image
            if isinstance(optionalimageinput, list):
                for img in optionalimageinput:
                    if isinstance(img, str) and img.strip():
                        imagesb64list.append(img.strip())
                        userimagesinfo.append(img.strip())
            elif isinstance(optionalimageinput, str) and optionalimageinput.strip():
                imagesb64list.append(optionalimageinput.strip())
                userimagesinfo.append(optionalimageinput.strip())
            elif ismultimodal:
                import numpy as np
                if hasattr(optionalimageinput, "numpy"):
                    arr = optionalimageinput.numpy()
                else:
                    arr = optionalimageinput
                if isinstance(arr, np.ndarray):
                    with io.BytesIO() as buf:
                        img = PIL.Image.fromarray(arr)
                        img.save(buf, format="PNG")
                        base64data = base64.b64encode(buf.getvalue()).decode("utf-8")
                        imagesb64list.append(base64data)
                        userimagesinfo.append("<image_tensor>")
        except Exception as e:
            self.logger.debug(f"AIO optional image processing error: {e}")

    def translateifneeded(self, text: str, target="EN", enable=True):
        if not enable or not text:
            return text
        try:
            lang_detected = self.detectlanguage(text)
            if lang_detected == target:
                return text
            deepl_api_key = os.environ.get("DEEPL_API_KEY", "").strip()
            if deepl_api_key:
                translated = self.translatedeepl(text, target, deepl_api_key)
                if translated:
                    return translated
            translated = self.translatelibres(text, target)
            if translated:
                return translated
        except Exception as e:
            self.logger.debug(f"Translation error: {e}")
        return text

    def detectlanguage(self, text: str):
        try:
            r = requests.post("https://libretranslate.com/detect", data={"q": text}, timeout=6).json()
            if isinstance(r, list) and len(r) > 0:
                return r[0].get("language", "en")
        except Exception:
            pass
        return "en"

    def translatedeepl(self, text: str, target="EN", apikey=""):
        try:
            if not apikey or not text:
                return None
            r = requests.post(
                "https://api-free.deepl.com/v2/translate",
                data={"auth_key": apikey, "text": text, "target_lang": target.upper()},
                timeout=8,
                headers={"User-Agent": ua()},
            )
            r.raise_for_status()
            data = r.json()
            if "translations" in data and data["translations"]:
                return data["translations"][0]["text"]
        except Exception:
            return None

    def translatelibres(self, text: str, target="EN"):
        try:
            r = requests.post(
                "https://libretranslate.com/translate",
                data={"q": text, "source": "auto", "target": target, "format": "text"},
                timeout=8,
                headers={"User-Agent": ua(), "Accept": "application/json"},
            )
            r.raise_for_status()
            return r.json().get("translatedText", None)
        except Exception:
            return None

    def buildkbindex(self, kbdir: str, chunkchars=900, overlapchars=120):
        files = sorted(
            glob.glob(os.path.join(kbdir, "**/*.txt"), recursive=True) +
            glob.glob(os.path.join(kbdir, "**/*.md"), recursive=True)
        )
        items = []
        for path in files:
            try:
                st = os.stat(path)
                items.append(f"{path}{int(st.st_mtime)}{st.st_size}")
            except Exception:
                continue
        sigstr = "".join(items) + f"{chunkchars}{overlapchars}"
        sig = hashlib.sha256(sigstr.encode("utf-8")).hexdigest()
        if sig == self.kbcachesig and self.kbready:
            return
        chunks = []
        for path in files:
            text = self.readtextfiles(path)
            if not text:
                continue
            chunks += self.chunktext(text, chunkchars, overlapchars)
        self.kbchunks = chunks
        self.kbready = True
        self.kbcachesig = sig

    def readtextfiles(self, path: str) -> Optional[str]:
        try:
            with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return None

    def chunktext(self, text: str, chunksize=900, overlap=120) -> list:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + chunksize, n)
            seg = text[i:end]
            m = re.search(r"[.!?]", seg[::-1])
            if m:
                pos = end - m.start()
                if pos - i > chunksize * 0.6:
                    end = pos
            chunk = text[i:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            i = max(end - overlap, i + 1)
        return chunks

    def ranktfidf(self, query: str, chunks: List[str], kbidf: Dict[str, float]):
        scored = []
        qtokens = self.tokenizes(query)
        qtf = Counter(qtokens)
        normqtf = {k: v / len(qtokens) for k, v in qtf.items()}
        for chunk in chunks:
            ctokens = self.tokenizes(chunk)
            tf = Counter(ctokens)
            denom = len(ctokens)
            tfidf_score = 0.0
            for term in qtokens:
                tf_term = tf.get(term, 0) / denom if denom > 0 else 0
                idf_term = kbidf.get(term, 0)
                tfidf_score += normqtf.get(term, 0) * tf_term * idf_term
            scored.append((chunk, tfidf_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, score) for chunk, score in scored[:10]]

    def collectliveonce(self, query: str, maxresults=5, timeout=10, wikilang="en", ddg=None):
        snippets = []
        sources = []
        if not query.strip():
            return snippets, sources
        if ddg is None:
            try:
                import duckduckgo_search
                ddg = duckduckgo_search.DuckDuckGoSearch()
            except ImportError:
                self.logger.debug("duckduckgo_search not installed")
                return snippets, sources
        try:
            abstxt = ddg.get_instant(query, timeout=timeout)
            absurl = ddg.get_instant_url(query, timeout=timeout)
            if abstxt:
                snippets.append(abstxt.strip())
            if absurl:
                sources.append(("duckduckgo", "Instant Answer", absurl))
            related = ddg.get_related(query, maxresults=maxresults, timeout=timeout)
            for rel in related:
                if isinstance(rel, dict) and "text" in rel and "url" in rel:
                    snippets.append(rel["text"])
                    sources.append(("duckduckgo", "Related", rel["url"]))
        except Exception as e:
            self.logger.debug(f"DuckDuckGo search failed: {e}")

        try:
            from wikipediaapi import Wikipedia
            wiki = Wikipedia(language=wikilang)
            page = wiki.page(query)
            if page.exists():
                summary = page.summary[0:1000]
                snippets.append(summary)
                sources.append(("wikipedia", "Page", page.fullurl))
        except Exception as e:
            self.logger.debug(f"Wikipedia search failed: {e}")

        return snippets, sources

    def sanitizemodeloutput(self, s: str):
        if not s:
            return s
        outlines = []
        rxsentence = re.compile(r'^[A-Z][a-z].*[\.\?\!]$')
        for raw in s.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("DEBUG") or stripped.lower().startswith("totalduration"):
                continue
            if rxsentence.match(stripped):
                continue
            if "Aristomenis Marinis presents" in stripped:
                continue
            outlines.append(line)
        normalized = "\n".join(outlines)
        normalized = re.sub(r"\n\s*\n", "\n\n", normalized)
        return normalized.strip()

    # More functions to follow...
    def harmonizeforgptossmodel(self, messages, enabled=True):
        if not enabled:
            return messages
        new_msgs = []
        for m in messages:
            if m.get("role") == "assistant":
                m["role"] = "system"
            new_msgs.append(m)
        return new_msgs

    def buildcontextblock(self, sources: List[Tuple[str, str, str]], snippets: List[str], maxcontextchars: int):
        lines = []
        lines.append("SECTION: Sources")
        for n, src in enumerate(sources, 1):
            title = src[1] if len(src) > 1 else ""
            url = src[2] if len(src) > 2 else ""
            lines.append(f"{n}. {title} - {url}")
        lines.append("SECTION: Snippets")
        for snippet in snippets:
            lines.append(snippet)
        content = "\n".join(lines)
        if len(content) > maxcontextchars:
            content = content[:maxcontextchars]
            content = content.rsplit(' ', 1)[0].strip()
        return content

    def looksnsfwurl(self, url: str):
        nsfw_patterns = [
            r"pornhub\.com",
            r"xvideos\.com",
            r"xnxx\.com",
            r"redtube\.com",
            r"xnxx",
            r"porn",
            r"xxx",
            r"adult",
        ]
        for pat in nsfw_patterns:
            if re.search(pat, url, re.IGNORECASE):
                return True
        return False

    def filternsfwurls(self, sources: List[Tuple[str, str, str]], allow_nsfw=False):
        if allow_nsfw:
            return sources
        filtered = []
        for src in sources:
            url = src[2] if len(src) > 2 else ""
            if not self.looksnsfwurl(url):
                filtered.append(src)
        return filtered

    def deduplicatesources(self, sources: List[Tuple[str, str, str]]):
        seen_urls = set()
        dedup = []
        for src in sources:
            url = src[2] if len(src) > 2 else ""
            if url and url not in seen_urls:
                dedup.append(src)
                seen_urls.add(url)
        return dedup

    def collectallcontextparallel(self, effectivequery: str, cfg: dict, livesnips: list,
                                  livesources: list, apisnips: list,
                                  imageitems: list, maxcontextchars=3600):
        weatherstruct = None
        if cfg.get("useweatherapis"):
            try:
                weatherstruct = self.getweatherstruct(effectivequery, timeout=10)
            except Exception as e:
                self.logger.debug(f"Weather struct error: {e}")

        if weatherstruct and cfg.get("useweatherapis"):
            apisnips.append(json.dumps({"Weather": self.compactweatherforcontext(weatherstruct)}))

        contextblock = self.buildcontextv3(
            kbhits=[],
            livesnips=livesnips,
            apisnips=apisnips,
            livesources=livesources,
            imageitems=imageitems,
            userimagesinfo=[],
            maxcontextchars=maxcontextchars,
        )

        if len(contextblock) > maxcontextchars:
            contextblock = contextblock[:maxcontextchars]

        return contextblock, weatherstruct

    def getweatherstruct(self, query: str, timeout=10):
        openmeteo_url = f"https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"
        try:
            r = requests.get(openmeteo_url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            current = data.get("current_weather", {})
            return current
        except Exception as e:
            self.logger.debug(f"OpenMeteo error: {e}")
            return None

    def compactweatherforcontext(self, ws: dict):
        parts = []
        if not ws:
            return ""
        if ws.get("temperature_2m") is not None:
            parts.append(f"Temperature: {ws['temperature_2m']}°C")
        if ws.get("windspeed_10m") is not None:
            parts.append(f"Windspeed: {ws['windspeed_10m']} km/h")
        if ws.get("weathercode") is not None:
            parts.append(f"Weather code: {ws['weathercode']}")
        return " | ".join(parts)

    def shutdown(self):
        self.logger.info("Shutting down OllamaSuperAIO...")

    # Node registration for ComfyUI
    NODECLASSMAPPINGS = {
        "AIOllama": "OllamaSuperAIO",
    }

    NODEDISPLAYNAMEMAPPINGS = {
        "AIOllama": "AIOllama Node",
    }

    def main():
        node = OllamaSuperAIO()
        if node:
            if node.selftest():
                print("AIOllama Node selftest passed!")
            else:
                print("AIOllama Node selftest failed.")

    def selftest(self):
        self.logger.info("Starting selftest...")
        try:
            models = self.bootstrapmodelsfordropdown()
            assert models, "No models loaded"
            self.logger.info(f"Available models: {models}")
        except Exception as e:
            self.logger.error(f"Selftest failed: {e}")
            return False
        return True

if __name__ == "__main__":
    main()

    def runregistrycategory(self, category, query, limit=3, timeout=8):
        items = []
        try:
            if category == "animals":
                url = f"https://catfact.ninja/fact"
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                fact = r.json().get("fact", "")
                items.append(fact)
            # Andere categorieën kunnen hier toegevoegd worden ...
        except Exception as e:
            self.logger.debug(f"Registry API error for {category}: {e}")
        return items[:limit]

    def collectallcontextparallel(self, effectivequery: str, cfg: dict, livesnips: list,
                                  livesources: list, apisnips: list,
                                  imageitems: list, maxcontextchars=3600):
        weatherstruct = None
        if cfg.get("useweatherapis"):
            try:
                weatherstruct = self.getweatherstruct(effectivequery, timeout=10)
            except Exception as e:
                self.logger.debug(f"Weather struct error: {e}")

        if weatherstruct and cfg.get("useweatherapis"):
            apisnips.append(json.dumps({"Weather": self.compactweatherforcontext(weatherstruct)}))

        contextblock = self.buildcontextv3(
            kbhits=[],
            livesnips=livesnips,
            apisnips=apisnips,
            livesources=livesources,
            imageitems=imageitems,
            userimagesinfo=[],
            maxcontextchars=maxcontextchars,
        )
        if len(contextblock) > maxcontextchars:
            contextblock = contextblock[:maxcontextchars]
        return contextblock, weatherstruct

    def getweatherstruct(self, query: str, timeout=10):
        openmeteo_url = f"https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"
        try:
            r = requests.get(openmeteo_url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            current = data.get("current_weather", {})
            return current
        except Exception as e:
            self.logger.debug(f"OpenMeteo error: {e}")
            return None

    def compactweatherforcontext(self, ws: dict):
        parts = []
        if not ws:
            return ""
        if ws.get("temperature_2m") is not None:
            parts.append(f"Temperature: {ws['temperature_2m']}°C")
        if ws.get("windspeed_10m") is not None:
            parts.append(f"Windspeed: {ws['windspeed_10m']} km/h")
        if ws.get("weathercode") is not None:
            parts.append(f"Weather code: {ws['weathercode']}")
        return " | ".join(parts)

    def looksnsfwurl(self, url: str):
        nsfw_patterns = [
            r"pornhub\.com",
            r"xvideos\.com",
            r"xnxx\.com",
            r"redtube\.com",
            r"xnxx",
            r"porn",
            r"xxx",
            r"adult",
        ]
        for pat in nsfw_patterns:
            if re.search(pat, url, re.IGNORECASE):
                return True
        return False

    def filternsfwurls(self, sources: List[Tuple[str, str, str]], allow_nsfw=False):
        if allow_nsfw:
            return sources
        filtered = []
        for src in sources:
            url = src[2] if len(src) > 2 else ""
            if not self.looksnsfwurl(url):
                filtered.append(src)
        return filtered

    def deduplicatesources(self, sources: List[Tuple[str, str, str]]):
        seen_urls = set()
        dedup = []
        for src in sources:
            url = src[2] if len(src) > 2 else ""
            if url and url not in seen_urls:
                dedup.append(src)
                seen_urls.add(url)
        return dedup

    def shutdown(self):
        self.logger.info("Shutting down OllamaSuperAIO...")

    # ComfyUI node registratie mappings
    NODECLASSMAPPINGS = {
        "AIOllama": "OllamaSuperAIO",
    }

    NODEDISPLAYNAMEMAPPINGS = {
        "AIOllama": "AIOllama Node",
    }

    def main():
        node = OllamaSuperAIO()
        if node:
            if node.selftest():
                print("AIOllama Node selftest passed!")
            else:
                print("AIOllama Node selftest failed.")

    def selftest(self):
        self.logger.info("Starting selftest...")
        try:
            models = self.bootstrapmodelsfordropdown()
            assert models, "No models loaded"
            self.logger.info(f"Available models: {models}")
        except Exception as e:
            self.logger.error(f"Selftest failed: {e}")
            return False
        return True

if __name__ == "__main__":
    main()

    def listregistrycategories(self):
        return [
            "animals", "books", "crypto", "devfun", "finance", "fx",
            "general", "geoip", "media", "music", "nature", "news",
            "sports", "anime"
        ]

    def generate(self, baseurl, model, messages, options=None, stream=False, keepalive=0):
        url = f"{baseurl.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keepalive": keepalive,
            **(options or {})
        }
        try:
            response = requests.post(url, json=payload, headers={"User-Agent": ua()}, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Generation API call failed: {e}")
            return None

    def harmonize_response(self, response):
        if not response:
            return None
        if isinstance(response, dict) and "choices" in response:
            text = response["choices"][0].get("message", {}).get("content", "")
            return text
        return response

    def reformat_thinking(self, text: str):
        thinking_pattern = re.compile(r"(?si)^think[^\w]*(.*)final[^\w]*(.*)", re.DOTALL | re.IGNORECASE)
        m = thinking_pattern.match(text)
        if m:
            think_output = m.group(1).strip()
            final_output = m.group(2).strip()
            return think_output, final_output
        return text, None

    def shutdown(self):
        self.logger.info("Shutting down OllamaSuperAIO...")

    def main():
        node = OllamaSuperAIO()
        if node:
            if node.selftest():
                print("AIOllama Node selftest passed!")
            else:
                print("AIOllama Node selftest failed.")

    if __name__ == "__main__":
        main()

    def processoptionalimageinput(self, optionalimageinput, ismultimodal, imagesb64list, userimagesinfo):
        if optionalimageinput is None:
            return
        try:
            import PIL.Image
            if isinstance(optionalimageinput, list):
                for img in optionalimageinput:
                    if isinstance(img, str) and img.strip():
                        imagesb64list.append(img.strip())
                        userimagesinfo.append(img.strip())
            elif isinstance(optionalimageinput, str) and optionalimageinput.strip():
                imagesb64list.append(optionalimageinput.strip())
                userimagesinfo.append(optionalimageinput.strip())
            elif ismultimodal:
                import numpy as np
                if hasattr(optionalimageinput, "numpy"):
                    arr = optionalimageinput.numpy()
                else:
                    arr = optionalimageinput
                if isinstance(arr, np.ndarray):
                    with io.BytesIO() as buf:
                        img = PIL.Image.fromarray(arr)
                        img.save(buf, format="PNG")
                        base64data = base64.b64encode(buf.getvalue()).decode("utf-8")
                        imagesb64list.append(base64data)
                        userimagesinfo.append("<image_tensor>")
        except Exception as e:
            self.logger.debug(f"AIO optional image processing error: {e}")

    def buildmessagessystemuser(self, systemprompt: str, userprompt: str, contextblock: str,
                                thinking: bool, multimodal: bool, imagesb64list: list,
                                answerlang: Optional[str] = None, contextmessagesjson: str = None):
        sys_prompt = (systemprompt or "").strip()
        if answerlang:
            sys_prompt += f"\nAnswer language: {answerlang}"
        messages = [{"role": "system", "content": sys_prompt}]
        user_content = userprompt.strip() + "\n\nContext:\n" + contextblock
        if contextmessagesjson:
            try:
                extra_messages = json.loads(contextmessagesjson)
                if isinstance(extra_messages, list):
                    messages.extend(extra_messages)
            except Exception:
                pass
        messages.append({"role": "user", "content": user_content})
        if thinking:
            messages.append({"role": "user", "content": "[Thinking enabled]"})
        if multimodal and imagesb64list:
            for img_b64 in imagesb64list:
                messages.append({"role": "user", "content": {"image": img_b64}})
        return messages

    def generate(self, baseurl, model, messages, options=None, stream=False, keepalive=0):
        url = f"{baseurl.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keepalive": keepalive,
            **(options or {})
        }
        try:
            response = requests.post(url, json=payload, headers={"User-Agent": ua()}, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Generation API call failed: {e}")
            return None

    def harmonize_response(self, response):
        if not response:
            return None
        if isinstance(response, dict) and "choices" in response:
            text = response["choices"][0].get("message", {}).get("content", "")
            return text
        return response

    def sanitizemodeloutput(self, s: str):
        if not s:
            return s
        outlines = []
        rxsentence = re.compile(r'^[A-Z][a-z].*[\.\?\!]$')
        for raw in s.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("DEBUG") or stripped.lower().startswith("totalduration"):
                continue
            if rxsentence.match(stripped):
                continue
            if "Aristomenis Marinis presents" in stripped:
                continue
            outlines.append(line)
        normalized = "\n".join(outlines)
        normalized = re.sub(r"\n\s*\n", "\n\n", normalized)
        return normalized.strip()

    def run(self, *args, **kwargs):
        self.logger.info("Starting run()")
        baseurl = kwargs.get("baseurl", "http://127.0.0.1:11434")
        model = kwargs.get("model", "llama3.1")
        userprompt = kwargs.get("userprompt", "")
        systemprompt = kwargs.get("systemprompt", "")
        multimodal = kwargs.get("multimodal", False)
        optionalimageinput = kwargs.get("optionalimageinput", None)
        maxcontextchars = kwargs.get("maxcontextchars", 3600)
        timeout = kwargs.get("timeout", 60)

        imagesb64list = []
        userimagesinfo = []
        ismultimodal = self.isprobablymultimodal(baseurl, model)
        self.processoptionalimageinput(optionalimageinput, ismultimodal, imagesb64list, userimagesinfo)

        contextblock, weather = self.collectallcontextparallel(
            effectivequery=userprompt,
            cfg=kwargs,
            livesnips=[],
            livesources=[],
            apisnips=[],
            imageitems=imagesb64list,
            maxcontextchars=maxcontextchars,
        )

        messages = self.buildmessagessystemuser(
            systemprompt=systemprompt,
            userprompt=userprompt,
            contextblock=contextblock,
            thinking=kwargs.get("thinking", False),
            multimodal=multimodal,
            imagesb64list=imagesb64list,
            answerlang=kwargs.get("answerlang", None),
            contextmessagesjson=kwargs.get("contextmessagesjson", None),
        )

        response = self.generate(baseurl, model, messages, stream=kwargs.get("stream", False), keepalive=kwargs.get("keepalive", 0))
        output = self.harmonize_response(response)
        finaloutput = self.sanitizemodeloutput(output)

        return finaloutput, json.dumps({"weather": weather}, indent=2)

    def buildcontextv3(self, kbhits, livesnips, apisnips, livesources, imageitems, userimagesinfo, maxcontextchars=3600):
        budgets = self.estimatecontextbudgets(maxcontextchars)
        apitakeweather = 650
        apitakeother = budgets.get('api', 2000) - apitakeweather
        context_lines = []

        if len(apisnips) > 0:
            apitake = []
            weatherlines = [s for s in apisnips if s.startswith('Weather')]
            otherlines = [s for s in apisnips if not s.startswith('Weather')]

            apitake += weatherlines[:apitakeweather]
            apitake += otherlines[:apitakeother]

            context_lines.extend(apitake)

        context_lines.extend(kbhits[:budgets.get('kb', 900)])
        context_lines.extend(livesnips[:budgets.get('live', 1000)])
        context_lines.extend(livesources[:budgets.get('live', 1000)])
        context_lines.extend(imageitems)
        context_lines.extend(userimagesinfo)

        joined_context = "\n".join(context_lines)
        if len(joined_context) > maxcontextchars:
            joined_context = joined_context[:maxcontextchars]
            joined_context = joined_context.rsplit(' ', 1)[0]
        return joined_context

    def estimatecontextbudgets(self, maxcontextchars):
        if maxcontextchars <= 4000:
            return {'kb': int(maxcontextchars * 0.40), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.18), 'images': int(maxcontextchars * 0.06),
                    'userimg': int(maxcontextchars * 0.06)}
        elif maxcontextchars <= 160000:
            return {'kb': int(maxcontextchars * 0.36), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.20), 'images': int(maxcontextchars * 0.07),
                    'userimg': int(maxcontextchars * 0.07)}
        else:
            return {'kb': int(maxcontextchars * 0.32), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.24), 'images': int(maxcontextchars * 0.07),
                    'userimg': int(maxcontextchars * 0.07)}

    def listregistrycategories(self):
        categories = [
            "animals",
            "books",
            "crypto",
            "devfun",
            "finance",
            "fx",
            "general",
            "geoip",
            "media",
            "music",
            "nature",
            "news",
            "sports",
            "anime",
        ]
        return categories

    def runregistrycategory(self, category, query, limit=3, timeout=8):
        items = []
        try:
            if category == "animals":
                url = f"https://catfact.ninja/fact"
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                fact = r.json().get("fact", "")
                items.append(fact)
            # Other categories ...
        except Exception as e:
            self.logger.debug(f"Registry API error for {category}: {e}")
        return items[:limit]

    def listollamamodels(self, baseurl, timeout=6):
        try:
            data = requests.get(f"{baseurl.rstrip('/')}/api/tags", headers={"User-Agent": ua()}, timeout=timeout).json()
            return [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
        except Exception as e:
            self.logger.debug(f"Failed to list models: {e}")
            return []

    def listregistrycategories(self):
        return [
            "animals", "books", "crypto", "devfun", "finance", "fx",
            "general", "geoip", "media", "music", "nature", "news",
            "sports", "anime"
        ]

    def runregistrycategory(self, category, query, limit=3, timeout=8):
        items = []
        try:
            if category == "animals":
                url = f"https://catfact.ninja/fact"
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                fact = r.json().get("fact", "")
                items.append(fact)
            # Add other categories as needed here
        except Exception as e:
            self.logger.debug(f"Registry API error for {category}: {e}")
        return items[:limit]

    def getweatherstruct(self, query: str, timeout=10):
        url = f"https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            current = data.get("current_weather", {})
            return current
        except Exception as e:
            self.logger.debug(f"OpenMeteo API error: {e}")
            return None

    def compactweatherforcontext(self, ws: dict):
        parts = []
        if not ws:
            return ""
        if ws.get("temperature_2m") is not None:
            parts.append(f"Temperature: {ws['temperature_2m']}°C")
        if ws.get("windspeed_10m") is not None:
            parts.append(f"Windspeed: {ws['windspeed_10m']} km/h")
        if ws.get("weathercode") is not None:
            parts.append(f"Weather code: {ws['weathercode']}")
        return " | ".join(parts)

    def harmonize_response(self, response):
        if not response:
            return None
        if isinstance(response, dict) and "choices" in response:
            choice = response["choices"][0]
            if isinstance(choice, dict):
                return choice.get("message", {}).get("content", "")
        return response

    def reformat_thinking(self, text: str):
        thinking_regex = re.compile(r"(?si)^think[^\w]*(.*)final[^\w]*(.*)", re.DOTALL | re.IGNORECASE)
        m = thinking_regex.match(text)
        if m:
            think_output = m.group(1).strip()
            final_output = m.group(2).strip()
            return think_output, final_output
        return text, None

    def selftest(self):
        self.logger.info("Starting selftest...")
        try:
            models = self.bootstrapmodelsfordropdown()
            assert models, "No models loaded"
            self.logger.info(f"Available models: {models}")
        except Exception as e:
            self.logger.error(f"Selftest failed: {e}")
            return False
        return True

    def shutdown(self):
        self.logger.info("Shutting down OllamaSuperAIO...")

# ComfyUI registration helpers mapping (to be integrated depending on usage)
NODECLASSMAPPINGS = {
    "AIOllama": OllamaSuperAIO,
}

NODEDISPLAYNAMEMAPPINGS = {
    "AIOllama": "AIOllama Node",
}

def main():
    node = OllamaSuperAIO()
    if node:
        if node.selftest():
            print("AIOllama Node selftest passed!")
        else:
            print("AIOllama Node selftest failed.")

if __name__ == "__main__":
    main()

    def buildmessagessystemuser(self, systemprompt: str, userprompt: str, contextblock: str,
                                thinking: bool, multimodal: bool, imagesb64list: list,
                                answerlang: Optional[str] = None, contextmessagesjson: str = None):
        sys_prompt = (systemprompt or "").strip()
        if answerlang:
            sys_prompt += f"\nAnswer language: {answerlang}"
        messages = [{"role": "system", "content": sys_prompt}]
        user_content = userprompt.strip() + "\n\nContext:\n" + contextblock
        if contextmessagesjson:
            try:
                extra_messages = json.loads(contextmessagesjson)
                if isinstance(extra_messages, list):
                    messages.extend(extra_messages)
            except Exception:
                pass
        messages.append({"role": "user", "content": user_content})
        if thinking:
            messages.append({"role": "user", "content": "[Thinking enabled]"})
        if multimodal and imagesb64list:
            for img_b64 in imagesb64list:
                messages.append({"role": "user", "content": {"image": img_b64}})
        return messages

    def shutdown(self):
        self.logger.info("Shutting down OllamaSuperAIO...")

# Node registration & execution

NODECLASSMAPPINGS = {
    "AIOllama": OllamaSuperAIO,
}

NODEDISPLAYNAMEMAPPINGS = {
    "AIOllama": "AIOllama Node",
}

def main():
    node = OllamaSuperAIO()
    if node.selftest():
        print("AIOllama Node selftest passed!")
    else:
        print("AIOllama Node selftest failed.")

if __name__ == "__main__":
    main()

# Additional helper functions that finalize context processing, filtering and managing sources

def looksnsfwurl(self, url: str):
    nsfw_patterns = [
        r"pornhub\.com", r"xvideos\.com", r"xnxx\.com", r"redtube\.com",
        r"xnxx", r"porn", r"xxx", r"adult",
    ]
    for pat in nsfw_patterns:
        if re.search(pat, url, re.IGNORECASE):
            return True
    return False

def filternsfwurls(self, sources: List[Tuple[str, str, str]], allow_nsfw=False):
    if allow_nsfw:
        return sources
    filtered = []
    for src in sources:
        url = src[2] if len(src) > 2 else ""
        if not self.looksnsfwurl(url):
            filtered.append(src)
    return filtered

def deduplicatesources(self, sources: List[Tuple[str, str, str]]):
    seen_urls = set()
    dedup = []
    for src in sources:
        url = src[2] if len(src) > 2 else ""
        if url and url not in seen_urls:
            dedup.append(src)
            seen_urls.add(url)
    return dedup

def buildcontextblock(self, sources: List[Tuple[str, str, str]], snippets: List[str], maxcontextchars: int):
    lines = []
    lines.append("SECTION: Sources")
    for n, src in enumerate(sources, 1):
        title = src[1] if len(src) > 1 else ""
        url = src[2] if len(src) > 2 else ""
        lines.append(f"{n}. {title} - {url}")
    lines.append("SECTION: Snippets")
    for snippet in snippets:
        lines.append(snippet)
    content = "\n".join(lines)
    if len(content) > maxcontextchars:
        content = content[:maxcontextchars]
        content = content.rsplit(' ', 1)[0].strip()
    return content

    def collectliveonce(self, query: str, maxresults=5, timeout=10, wikilang="en", ddg=None):
        snippets = []
        sources = []
        if not query.strip():
            return snippets, sources
        if ddg is None:
            try:
                import duckduckgo_search
                ddg = duckduckgo_search.DuckDuckGoSearch()
            except ImportError:
                self.logger.debug("duckduckgo_search library not installed")
                return snippets, sources
        try:
            abstxt = ddg.get_instant(query, timeout=timeout)
            absurl = ddg.get_instant_url(query, timeout=timeout)
            if abstxt:
                snippets.append(abstxt.strip())
            if absurl:
                sources.append(("duckduckgo", "Instant Answer", absurl))
            related = ddg.get_related(query, maxresults=maxresults, timeout=timeout)
            for rel in related:
                if isinstance(rel, dict) and "text" in rel and "url" in rel:
                    snippets.append(rel["text"])
                    sources.append(("duckduckgo", "Related", rel["url"]))
        except Exception as e:
            self.logger.debug(f"DuckDuckGo search failed: {e}")

        try:
            from wikipediaapi import Wikipedia
            wiki = Wikipedia(language=wikilang)
            page = wiki.page(query)
            if page.exists():
                summary = page.summary[0:1000]
                snippets.append(summary)
                sources.append(("wikipedia", "Page", page.fullurl))
        except Exception as e:
            self.logger.debug(f"Wikipedia search failed: {e}")

        return snippets, sources

    def run(self, *args, **kwargs):
        self.logger.info("Starting run()")
        baseurl = kwargs.get("baseurl", "http://127.0.0.1:11434")
        model = kwargs.get("model", "llama3.1")
        userprompt = kwargs.get("userprompt", "")
        systemprompt = kwargs.get("systemprompt", "")
        multimodal = kwargs.get("multimodal", False)
        optionalimageinput = kwargs.get("optionalimageinput", None)
        maxcontextchars = kwargs.get("maxcontextchars", 3600)
        timeout = kwargs.get("timeout", 60)

        imagesb64list = []
        userimagesinfo = []
        ismultimodal = self.isprobablymultimodal(baseurl, model)
        self.processoptionalimageinput(optionalimageinput, ismultimodal, imagesb64list, userimagesinfo)

        contextblock, weather = self.collectallcontextparallel(
            effectivequery=userprompt,
            cfg=kwargs,
            livesnips=[],
            livesources=[],
            apisnips=[],
            imageitems=imagesb64list,
            maxcontextchars=maxcontextchars,
        )

        messages = self.buildmessagessystemuser(
            systemprompt=systemprompt,
            userprompt=userprompt,
            contextblock=contextblock,
            thinking=kwargs.get("thinking", False),
            multimodal=multimodal,
            imagesb64list=imagesb64list,
            answerlang=kwargs.get("answerlang", None),
            contextmessagesjson=kwargs.get("contextmessagesjson", None),
        )

        response = self.generate(baseurl, model, messages, stream=kwargs.get("stream", False), keepalive=kwargs.get("keepalive", 0))
        output = self.harmonize_response(response)
        finaloutput = self.sanitizemodeloutput(output)

        return finaloutput, json.dumps({"weather": weather}, indent=2)

    def estimatecontextbudgets(self, maxcontextchars):
        if maxcontextchars <= 4000:
            return {'kb': int(maxcontextchars * 0.40), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.18), 'images': int(maxcontextchars * 0.06),
                    'userimg': int(maxcontextchars * 0.06)}
        elif maxcontextchars <= 160000:
            return {'kb': int(maxcontextchars * 0.36), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.20), 'images': int(maxcontextchars * 0.07),
                    'userimg': int(maxcontextchars * 0.07)}
        else:
            return {'kb': int(maxcontextchars * 0.32), 'live': int(maxcontextchars * 0.30),
                    'api': int(maxcontextchars * 0.24), 'images': int(maxcontextchars * 0.07),
                    'userimg': int(maxcontextchars * 0.07)}

    def buildcontextv3(self, kbhits, livesnips, apisnips, livesources, imageitems, userimagesinfo, maxcontextchars=3600):
        budgets = self.estimatecontextbudgets(maxcontextchars)
        apitakeweather = 650
        apitakeother = budgets.get('api', 2000) - apitakeweather
        context_lines = []

        if len(apisnips) > 0:
            apitake = []
            weatherlines = [s for s in apisnips if s.startswith('Weather')]
            otherlines = [s for s in apisnips if not s.startswith('Weather')]

            apitake += weatherlines[:apitakeweather]
            apitake += otherlines[:apitakeother]

            context_lines.extend(apitake)

        context_lines.extend(kbhits[:budgets.get('kb', 900)])
        context_lines.extend(livesnips[:budgets.get('live', 1000)])
        context_lines.extend(livesources[:budgets.get('live', 1000)])
        context_lines.extend(imageitems)
        context_lines.extend(userimagesinfo)

        joined_context = "\n".join(context_lines)
        if len(joined_context) > maxcontextchars:
            joined_context = joined_context[:maxcontextchars]
            joined_context = joined_context.rsplit(' ', 1)[0]
        return joined_context

    def processoptionalimageinput(self, optionalimageinput, ismultimodal, imagesb64list, userimagesinfo):
        if optionalimageinput is None:
            return
        try:
            import PIL.Image
            if isinstance(optionalimageinput, list):
                for img in optionalimageinput:
                    if isinstance(img, str) and img.strip():
                        imagesb64list.append(img.strip())
                        userimagesinfo.append(img.strip())
            elif isinstance(optionalimageinput, str) and optionalimageinput.strip():
                imagesb64list.append(optionalimageinput.strip())
                userimagesinfo.append(optionalimageinput.strip())
            elif ismultimodal:
                import numpy as np
                if hasattr(optionalimageinput, "numpy"):
                    arr = optionalimageinput.numpy()
                else:
                    arr = optionalimageinput
                if isinstance(arr, np.ndarray):
                    with io.BytesIO() as buf:
                        img = PIL.Image.fromarray(arr)
                        img.save(buf, format="PNG")
                        base64data = base64.b64encode(buf.getvalue()).decode("utf-8")
                        imagesb64list.append(base64data)
                        userimagesinfo.append("<image_tensor>")
        except Exception as e:
            self.logger.debug(f"AIO optional image processing error: {e}")

    def tokenizes(self, s: str) -> List[str]:
        # Simple tokenizer: lowercase, alphanumerics only
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", s)]

    def ranktfidf(self, query: str, chunks: List[str], kbidf: Dict[str, float]) -> List[Tuple[str, float]]:
        scored = []
        qtokens = self.tokenizes(query)
        qtf = Counter(qtokens)
        normqtf = {k: v / len(qtokens) for k, v in qtf.items()}
        for chunk in chunks:
            ctokens = self.tokenizes(chunk)
            tf = Counter(ctokens)
            denom = len(ctokens)
            tfidf_score = 0.0
            for term in qtokens:
                tf_term = tf.get(term, 0) / denom if denom > 0 else 0
                idf_term = kbidf.get(term, 0)
                tfidf_score += normqtf.get(term, 0) * tf_term * idf_term
            scored.append((chunk, tfidf_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:10]

    def tokenizes(self, s: str) -> list:
        # Simple tokenizer: lowercase words and digits only
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", s)]

    def ranktfidf(self, query: str, chunks: list, kbidf: dict) -> list:
        scored = []
        qtokens = self.tokenizes(query)
        qtf = Counter(qtokens)
        normqtf = {k: v / len(qtokens) for k, v in qtf.items()}
        for chunk in chunks:
            ctokens = self.tokenizes(chunk)
            tf = Counter(ctokens)
            denom = len(ctokens)
            tfidf_score = 0.0
            for term in qtokens:
                tf_term = tf.get(term, 0) / denom if denom > 0 else 0
                idf_term = kbidf.get(term, 0)
                tfidf_score += normqtf.get(term, 0) * tf_term * idf_term
            scored.append((chunk, tfidf_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, score) for chunk, score in scored[:10]]

    def buildcontextblock(self, sources: List[Tuple[str, str, str]], snippets: List[str], maxcontextchars: int):
        lines = []
        lines.append("SECTION: Sources")
        for n, src in enumerate(sources, 1):
            title = src[1] if len(src) > 1 else ""
            url = src[2] if len(src) > 2 else ""
            lines.append(f"{n}. {title} - {url}")
        lines.append("SECTION: Snippets")
        for snippet in snippets:
            lines.append(snippet)
        content = "\n".join(lines)
        if len(content) > maxcontextchars:
            content = content[:maxcontextchars]
            content = content.rsplit(' ', 1)[0].strip()
        return content

    def runregistrycategory(self, category, query, limit=3, timeout=8):
        items = []
        try:
            if category == "animals":
                url = "https://catfact.ninja/fact"
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                fact = r.json().get("fact", "")
                items.append(fact)
            # Andere categorieën kunnen hier toegevoegd worden
        except Exception as e:
            self.logger.debug(f"Registry API-error voor categorie {category}: {e}")
        return items[:limit]

    def listregistrycategories(self):
        return [
            "animals", "books", "crypto", "devfun", "finance", "fx",
            "general", "geoip", "media", "music", "nature", "news",
            "sports", "anime"
        ]

    def getweatherstruct(self, query: str, timeout=10):
        try:
            url = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("current_weather", {})
        except Exception as e:
            self.logger.debug(f"OpenMeteo API error: {e}")
            return None

    def compactweatherforcontext(self, ws: dict):
        parts = []
        if not ws:
            return ""
        if "temperature_2m" in ws:
            parts.append(f"Temperature: {ws['temperature_2m']}°C")
        if "windspeed_10m" in ws:
            parts.append(f"Windspeed: {ws['windspeed_10m']} km/h")
        if "weathercode" in ws:
            parts.append(f"Weather code: {ws['weathercode']}")
        return " | ".join(parts)

    def sanitizemodeloutput(self, s: str) -> str:
        if not s:
            return s
        outlines = []
        rxsentence = re.compile(r'^[A-Z][a-z].*[\.\?\!]$')
        for raw in s.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            # Filter debug or unrelated lines
            if stripped.startswith("DEBUG") or stripped.lower().startswith("totalduration"):
                continue
            if rxsentence.match(stripped):
                continue
            if "Aristomenis Marinis presents" in stripped:
                continue
            outlines.append(line)
        normalized = "\n".join(outlines)
        normalized = re.sub(r"\n\s*\n", "\n\n", normalized)
        return normalized.strip()

    def reformat_thinking(self, text: str) -> tuple:
        thinking_regex = re.compile(r"(?si)^think[^\w]*(.*)final[^\w]*(.*)", re.DOTALL | re.IGNORECASE)
        m = thinking_regex.match(text)
        if m:
            think_output = m.group(1).strip()
            final_output = m.group(2).strip()
            return think_output, final_output
        return text, None

    def harmonizeforgptossmodel(self, messages, enabled=True):
        if not enabled:
            return messages
        new_msgs = []
        for m in messages:
            if m.get("role") == "assistant":
                m["role"] = "system"
            new_msgs.append(m)
        return new_msgs

    def healthcheck(self, baseurl: str, timeout=6):
        try:
            r = requests.get(f"{baseurl.rstrip('/')}/api/tags", headers={"User-Agent": ua()}, timeout=timeout)
            r.raise_for_status()
            return True
        except Exception as e:
            self.logger.debug(f"AIO Health check failed: {e}")
            return False

    def withretry(self, func, tries=2, delay=0.35, backoff=1.6, jitter=0.1, *args, **kwargs):
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

    def cacheget(self, key):
        rec = self.ttlcache.get(key)
        if not rec:
            return None
        ts, val, ttl = rec
        if time.time() - ts > ttl:
            self.ttlcache.pop(key, None)
            return None
        return val

    def cacheput(self, key, value, ttl=None):
        self.ttlcache[key] = (time.time(), value, ttl or self.ttldefault)

    def cachekey(self, name, params):
        try:
            return name + hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        except Exception:
            return name + str(params)

    def withretry(self, func, tries=2, delay=0.35, backoff=1.6, jitter=0.1, *args, **kwargs):
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

    def translatedeepl(self, text: str, target="EN", apikey=""):
        try:
            if not apikey or not text:
                return None
            r = requests.post(
                "https://api-free.deepl.com/v2/translate",
                data={"auth_key": apikey, "text": text, "target_lang": target.upper()},
                timeout=8,
                headers={"User-Agent": ua()},
            )
            r.raise_for_status()
            data = r.json()
            if "translations" in data and data["translations"]:
                return data["translations"][0]["text"]
        except Exception:
            return None

    def translatelibres(self, text: str, target="EN"):
        try:
            r = requests.post(
                "https://libretranslate.com/translate",
                data={"q": text, "source": "auto", "target": target, "format": "text"},
                timeout=8,
                headers={"User-Agent": ua(), "Accept": "application/json"},
            )
            r.raise_for_status()
            return r.json().get("translatedText", None)
        except Exception:
            return None

    def detectlanguage(self, text: str):
        try:
            r = requests.post("https://libretranslate.com/detect", data={"q": text}, timeout=6).json()
            if isinstance(r, list) and len(r) > 0:
                return r[0].get("language", "en")
        except Exception:
            pass
        return "en"

    def translateifneeded(self, text: str, target="EN", enable=True):
        if not enable or not text:
            return text
        try:
            lang_detected = self.detectlanguage(text)
            if lang_detected == target:
                return text
            deepl_api_key = os.environ.get("DEEPL_API_KEY", "").strip()
            if deepl_api_key:
                translated = self.translatedeepl(text, target, deepl_api_key)
                if translated:
                    return translated
            translated = self.translatelibres(text, target)
            if translated:
                return translated
        except Exception as e:
            self.logger.debug(f"Translation error: {e}")
        return text

    def detectlanguage(self, text: str):
        try:
            r = requests.post("https://libretranslate.com/detect", data={"q": text}, timeout=6).json()
            if isinstance(r, list) and len(r) > 0:
                return r[0].get("language", "en")
        except Exception:
            pass
        return "en"

    def translatelibres(self, text: str, target="EN"):
        try:
            r = requests.post(
                "https://libretranslate.com/translate",
                data={"q": text, "source": "auto", "target": target, "format": "text"},
                timeout=8,
                headers={"User-Agent": ua(), "Accept": "application/json"},
            )
            r.raise_for_status()
            return r.json().get("translatedText", None)
        except Exception:
            return None

def ua():
    # Returns user agent string for HTTP requests
    return "Mozilla/5.0 (compatible; AIOllamaNode/1.0; +https://example.com)"

    def looksnsfwurl(self, url: str) -> bool:
        nsfw_patterns = [
            r"pornhub\.com",
            r"xvideos\.com",
            r"xnxx\.com",
            r"redtube\.com",
            r"xnxx",
            r"porn",
            r"xxx",
            r"adult",
        ]
        for pat in nsfw_patterns:
            if re.search(pat, url, re.IGNORECASE):
                return True
        return False

    def filternsfwurls(self, sources: list, allow_nsfw=False) -> list:
        if allow_nsfw:
            return sources
        filtered = []
        for src in sources:
            url = src[2] if len(src) > 2 else ""
            if not self.looksnsfwurl(url):
                filtered.append(src)
        return filtered

    def deduplicatesources(self, sources: list) -> list:
        seen_urls = set()
        dedup = []
        for src in sources:
            url = src[2] if len(src) > 2 else ""
            if url and url not in seen_urls:
                dedup.append(src)
                seen_urls.add(url)
        return dedup

    def collectallcontextparallel(self, effectivequery, cfg, livesnips, livesources, apisnips, imageitems, maxcontextchars):
        # For now, directly build context block with given snippets and sources
        all_snips = livesnips + apisnips
        sources = livesources
        contextblock = self.buildcontextv3(
            kbhits=all_snips,
            livesnips=livesnips,
            apisnips=apisnips,
            livesources=livesources,
            imageitems=imageitems,
            userimagesinfo=[],
            maxcontextchars=maxcontextchars
        )
        # return context block and dummy weather info
        return contextblock, {}

# Additional utility functions or end-of-file markers could be here,
# but the previous blocks covered almost all functionality of the AIOllama node.

# Example snippet:
# def noop(self):
#     pass

# If requested, everything can be integrated into a single file for ease of use.
