"""
Combined Ultimate Website Analyzer API v4.1 - PERFORMANCE OPTIMIZED
Optimized for handling multiple concurrent requests and fast sequential scans
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any, Union
import asyncio
import aiohttp
import httpx
import time
import os
import logging
from datetime import datetime
from cachetools import TTLCache
import json
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import ssl
import socket
from dataclasses import dataclass
import hashlib
from playwright.async_api import async_playwright
from collections import defaultdict, Counter
import math
from enum import Enum
import weakref
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = os.getenv("API_VERSION", "4.1.0")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))  # Increased workers
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "120"))  # Reduced timeout

# Google API Keys for real Core Web Vitals
PSI_KEY = os.getenv("PSI_API_KEY")
CRUX_KEY = os.getenv("CRUX_API_KEY")

# Google API URLs
PSI_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
CRUX_URL = "https://chromeuxreport.googleapis.com/v1/records:query"

# Optimized cache settings - shorter TTL for better memory management
VITALS_CACHE = TTLCache(maxsize=500, ttl=60*30)  # 30 minutes instead of 12 hours

app = FastAPI(
    title="Combined Ultimate Website Analyzer API (OPTIMIZED)",
    description="High-performance website analysis optimized for multiple sequential scans",
    version=API_VERSION,
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None
)

# Add security middleware for production
if ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*.onrender.com", "localhost"]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENVIRONMENT == "development" else ["https://*.onrender.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request logging middleware with performance tracking
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.2f}s")
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {str(e)} - {process_time:.2f}s")
        raise

# =====================
# MODELS (unchanged)
# =====================

class Device(str, Enum):
    desktop = "desktop"
    mobile = "mobile"

class ScanMode(str, Enum):
    quick = "quick"
    standard = "standard"
    comprehensive = "comprehensive"

class CombinedAnalyzeRequest(BaseModel):
    url: HttpUrl
    mode: ScanMode = ScanMode.comprehensive
    device: Device = Device.desktop
    include_performance: bool = True
    include_real_vitals: bool = True
    include_seo: bool = True
    include_security: bool = True
    include_content: bool = True
    include_images: bool = True
    include_business_info: bool = True
    include_technical: bool = True
    include_structured_data: bool = True
    include_accessibility: bool = True
    include_mobile: bool = True
    include_external_resources: bool = True
    use_origin_fallback: bool = True

# =====================
# OPTIMIZED BROWSER MANAGER
# =====================

class OptimizedBrowserManager:
    """High-performance browser manager with pooling and resource optimization"""
    
    def __init__(self, pool_size=3):
        self.pool_size = pool_size
        self.browser_pool = []
        self.context_pool = []
        self.playwright = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._cleanup_interval = 60  # Cleanup every minute
        self._last_cleanup = time.time()
        
    async def initialize(self):
        """Initialize browser pool"""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            try:
                self.playwright = await async_playwright().start()
                
                # Create browser pool
                for i in range(self.pool_size):
                    browser = await self.playwright.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-gpu',
                            '--no-zygote',
                            '--disable-blink-features=AutomationControlled',
                            '--memory-pressure-off',
                            '--max-old-space-size=256',  # Reduced memory
                            '--disable-background-timer-throttling',
                            '--disable-renderer-backgrounding',
                            '--disable-backgrounding-occluded-windows',
                            '--disable-component-extensions-with-background-pages',
                            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        ]
                    )
                    self.browser_pool.append({
                        'browser': browser,
                        'in_use': False,
                        'created_at': time.time()
                    })
                
                self._initialized = True
                logger.info(f"Browser pool initialized with {self.pool_size} browsers")
                
            except Exception as e:
                logger.error(f"Failed to initialize browser pool: {e}")
                raise
    
    async def get_browser_context(self, viewport=None, user_agent=None):
        """Get browser context from pool with automatic cleanup"""
        await self.initialize()
        
        # Periodic cleanup
        if time.time() - self._last_cleanup > self._cleanup_interval:
            await self._cleanup_resources()
        
        async with self._lock:
            # Find available browser
            for browser_info in self.browser_pool:
                if not browser_info['in_use']:
                    browser_info['in_use'] = True
                    try:
                        context = await browser_info['browser'].new_context(
                            viewport=viewport or {'width': 1366, 'height': 768},
                            user_agent=user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            extra_http_headers={
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                'Accept-Language': 'en-US,en;q=0.5',
                                'Cache-Control': 'no-cache'
                            },
                            # Optimize for speed
                            java_script_enabled=True,
                            ignore_https_errors=True,
                        )
                        
                        return context, browser_info
                        
                    except Exception as e:
                        browser_info['in_use'] = False
                        logger.warning(f"Failed to create context: {e}")
                        continue
            
            # All browsers busy - create temporary one
            logger.warning("All browsers busy, creating temporary browser")
            browser = await self.playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport=viewport or {'width': 1366, 'height': 768}
            )
            return context, {'browser': browser, 'in_use': False, 'temporary': True}
    
    async def release_context(self, context, browser_info):
        """Release browser context back to pool"""
        try:
            if context and not context.is_closed:
                await context.close()
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
        
        # Handle temporary browsers
        if browser_info.get('temporary'):
            try:
                await browser_info['browser'].close()
            except:
                pass
        else:
            browser_info['in_use'] = False
    
    async def _cleanup_resources(self):
        """Cleanup stale resources"""
        try:
            # Force garbage collection
            gc.collect()
            self._last_cleanup = time.time()
            logger.debug("Resource cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    async def close(self):
        """Close all browsers and cleanup"""
        if not self._initialized:
            return
            
        async with self._lock:
            for browser_info in self.browser_pool:
                try:
                    await browser_info['browser'].close()
                except:
                    pass
            
            if self.playwright:
                try:
                    await self.playwright.stop()
                except:
                    pass
            
            self.browser_pool.clear()
            self._initialized = False
            logger.info("Browser pool closed")

# Global browser manager
browser_manager = OptimizedBrowserManager(pool_size=3)

# =====================
# GOOGLE API UTILITIES (optimized)
# =====================

def ms_to_s(v: Optional[float]) -> Optional[float]:
    """Convert milliseconds to seconds with 3 decimal precision"""
    return round(v/1000.0, 3) if isinstance(v, (int, float)) else None

def get_rating(metric: str, value: Optional[float]) -> Optional[str]:
    """Get rating label for Core Web Vitals metrics"""
    if value is None: 
        return None
    
    thresholds = {
        "LCP": (2.5, 4.0),
        "INP": (0.2, 0.5),
        "CLS": (0.1, 0.25),
        "FID": (0.1, 0.3)
    }
    
    if metric not in thresholds:
        return None
        
    good_threshold, poor_threshold = thresholds[metric]
    
    if value <= good_threshold:
        return "good"
    elif value <= poor_threshold:
        return "needs_improvement"
    else:
        return "poor"

async def get_crux_data(url: str, form_factor: str = "PHONE", use_origin_fallback: bool = True) -> Dict[str, Any]:
    """Get real CrUX field data with optimized timeouts"""
    if not CRUX_KEY:
        return {"available": False, "reason": "CRUX_API_KEY not configured"}

    payload = {"url": url, "formFactor": form_factor}
    headers = {"Content-Type": "application/json"}

    # Reduced timeout for faster response
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            response = await client.post(
                f"{CRUX_URL}?key={CRUX_KEY}", 
                json=payload, 
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                metrics = data.get("record", {}).get("metrics", {})
                if metrics:
                    return await parse_crux_metrics(metrics, "page")
        except Exception as e:
            logger.warning(f"CrUX page-level error: {e}")

        # Fallback to origin-level data
        if use_origin_fallback:
            try:
                parsed = urlparse(url)
                origin = f"{parsed.scheme}://{parsed.netloc}"
                
                origin_payload = {"origin": origin, "formFactor": form_factor}
                
                response = await client.post(
                    f"{CRUX_URL}?key={CRUX_KEY}", 
                    json=origin_payload, 
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    metrics = data.get("record", {}).get("metrics", {})
                    if metrics:
                        return await parse_crux_metrics(metrics, "origin")
            except Exception as e:
                logger.warning(f"CrUX origin-level error: {e}")

    return {"available": False, "reason": "No CrUX data available"}

async def parse_crux_metrics(metrics: Dict, source: str) -> Dict[str, Any]:
    """Parse CrUX metrics and convert to seconds where needed"""
    
    def get_p75(metric_key: str, convert_to_seconds: bool = True) -> Optional[float]:
        try:
            value = metrics.get(metric_key, {}).get("percentiles", {}).get("p75")
            if value is None:
                return None
            return ms_to_s(value) if convert_to_seconds else value
        except:
            return None

    return {
        "available": True,
        "source": source,
        "LCP_s": get_p75("largest_contentful_paint", True),
        "INP_s": get_p75("interaction_to_next_paint", True),
        "CLS": get_p75("cumulative_layout_shift", False),
        "FID_s": get_p75("first_input_delay", True)
    }

async def get_psi_data(url: str, strategy: str = "mobile") -> Dict[str, Any]:
    """Get real PageSpeed Insights data with optimized timeout"""
    if not PSI_KEY:
        return {"available": False, "reason": "PSI_API_KEY not configured"}

    params = {
        "url": url,
        "strategy": strategy,
        "category": "performance",
        "key": PSI_KEY
    }

    # Reduced timeout for faster response
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            response = await client.get(PSI_URL, params=params)
            
            if response.status_code != 200:
                return {
                    "available": False, 
                    "reason": f"PSI API error: {response.status_code}",
                    "details": response.text[:200]
                }
            
            data = response.json()
            lighthouse_result = data.get("lighthouseResult", {})
            audits = lighthouse_result.get("audits", {})
            
            perf_score = lighthouse_result.get("categories", {}).get("performance", {}).get("score")
            if isinstance(perf_score, (int, float)):
                perf_score = round(perf_score * 100)

            lcp_audit = audits.get("largest-contentful-paint", {})
            cls_audit = audits.get("cumulative-layout-shift", {})
            inp_audit = audits.get("interaction-to-next-paint", {})
            
            return {
                "available": True,
                "LCP_s": ms_to_s(lcp_audit.get("numericValue")),
                "CLS": cls_audit.get("numericValue"),
                "INP_s": ms_to_s(inp_audit.get("numericValue")),
                "performance_score": perf_score,
                "loading_experience": data.get("loadingExperience", {}).get("overall_category"),
                "origin_fallback": data.get("originFallback", False)
            }
            
        except Exception as e:
            logger.warning(f"PSI request failed: {str(e)}")
            return {"available": False, "reason": f"PSI request failed: {str(e)}"}

def merge_field_and_lab_data(field_data: Dict[str, Any], lab_data: Dict[str, Any]) -> Dict[str, Any]:
    """Combine field data (CrUX) and lab data (PSI) with field data priority"""
    lcp = field_data.get("LCP_s") if field_data.get("LCP_s") is not None else lab_data.get("LCP_s")
    inp = field_data.get("INP_s") if field_data.get("INP_s") is not None else lab_data.get("INP_s")
    cls = field_data.get("CLS") if field_data.get("CLS") is not None else lab_data.get("CLS")
    
    return {
        "LCP_s": lcp,
        "INP_s": inp,
        "CLS": cls,
        "FID_s": field_data.get("FID_s"),
        "ratings": {
            "LCP": get_rating("LCP", lcp),
            "INP": get_rating("INP", inp),
            "CLS": get_rating("CLS", cls)
        }
    }

async def get_real_core_web_vitals(url: str, device: str = "mobile", use_origin_fallback: bool = True) -> Dict[str, Any]:
    """Get REAL Core Web Vitals from Google APIs with caching"""
    # Cache check
    cache_key = (url, device)
    if cache_key in VITALS_CACHE:
        cached_result = VITALS_CACHE[cache_key].copy()
        cached_result["from_cache"] = True
        return cached_result

    form_factor = "PHONE" if device.lower() == "mobile" else "DESKTOP"
    strategy = "mobile" if device.lower() == "mobile" else "desktop"
    
    # Get data (parallel for speed)
    field_data, lab_data = await asyncio.gather(
        get_crux_data(url, form_factor, use_origin_fallback),
        get_psi_data(url, strategy)
    )
    
    combined_metrics = merge_field_and_lab_data(field_data, lab_data)
    
    result = {
        "url": url,
        "device": device,
        "timestamp": int(time.time()),
        "field_data": {
            "available": field_data.get("available", False),
            "source": field_data.get("source"),
            "reason": field_data.get("reason") if not field_data.get("available") else None
        },
        "lab_data": {
            "available": lab_data.get("available", False),
            "performance_score": lab_data.get("performance_score"),
            "reason": lab_data.get("reason") if not lab_data.get("available") else None
        },
        "metrics": combined_metrics,
        "from_cache": False
    }
    
    # Cache the result
    VITALS_CACHE[cache_key] = result
    return result

# =====================
# OPTIMIZED ANALYZER SERVICE
# =====================

class OptimizedCombinedWebsiteAnalyzer:
    """Optimized analyzer for high-performance sequential scans"""
    
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    async def analyze_website(self, url: str, options: CombinedAnalyzeRequest) -> Dict[str, Any]:
        """Optimized main analysis function"""
        start_time = time.time()
        context = None
        browser_info = None
        
        try:
            logger.info(f"Starting optimized analysis for: {url}")
            
            # Get browser context from pool
            viewport = {
                'width': 1920 if options.device == Device.desktop else 390, 
                'height': 1080 if options.device == Device.desktop else 844
            }
            context, browser_info = await browser_manager.get_browser_context(viewport=viewport)
            page = await context.new_page()
            
            # Optimized navigation with progressive timeouts
            response = await self._navigate_with_retry(page, url)
            
            # Shorter wait time for faster processing
            await page.wait_for_timeout(2000)  # Reduced from 3000ms
            
            # Get HTML content
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Build result structure
            result = {
                "url": url,
                "final_url": page.url,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": 0,
                "status_code": response.status if response else None,
                "api_version": API_VERSION
            }
            
            # ==========================================
            # OPTIMIZED PARALLEL ANALYSIS
            # ==========================================
            
            # Group lightweight analyses together
            lightweight_tasks = []
            
            if options.include_performance:
                lightweight_tasks.append(self._analyze_performance(page, soup, response))
            if options.include_seo:
                lightweight_tasks.append(self._analyze_seo_structured(soup, url, response))
            if options.include_security:
                lightweight_tasks.append(self._analyze_security(page.url, response))
            
            # Execute lightweight tasks in parallel
            if lightweight_tasks:
                performance_result, seo_result, security_result = await asyncio.gather(*lightweight_tasks)
                result["performance"] = performance_result
                result["seo_analysis"] = seo_result
                result["security_analysis"] = security_result
            
            # Content analysis (fast, synchronous)
            result["content_analysis"] = await self._analyze_content_structured(soup)
            
            # Conditional analyses based on options
            if options.include_images:
                result["images_analysis"] = await self._analyze_images_structured(soup, page.url)
            
            if options.include_technical:
                result["technical_analysis"] = await self._analyze_technical_structured(page, soup, response)
            
            # Links analysis
            result["links_analysis"] = await self._analyze_links_structured(soup, page.url)
            
            if options.include_structured_data:
                result["structured_data"] = await self._analyze_structured_data(soup)
            
            if options.include_mobile:
                result["mobile_analysis"] = await self._analyze_mobile(soup)
            
            if options.include_accessibility:
                result["accessibility_analysis"] = await self._analyze_accessibility(soup)
            
            # ==========================================
            # DETAILED ANALYSIS (optimized)
            # ==========================================
            
            result["page_info"] = self._collect_page_info(page.url, soup)
            
            if options.include_business_info:
                result["business_info"] = await self._collect_business_info(soup, page.url)
            
            result["contact_info"] = await self._collect_contact_info(soup)
            
            if options.include_content:
                result["content"] = await self._collect_enhanced_content(soup)
            
            result["links"] = await self._collect_enhanced_links(soup, page.url)
            
            if options.include_images:
                result["images"] = await self._collect_enhanced_images(soup, page.url)
            
            result["meta_data"] = await self._collect_meta_data(soup)
            result["page_structure"] = await self._collect_page_structure(soup)
            result["seo"] = await self._collect_enhanced_seo(soup, page.url)
            result["social_media"] = await self._collect_social_media(soup)
            result["technical"] = await self._collect_technical_detailed(page, soup, response)
            
            # External resources (lightweight)
            if options.include_external_resources:
                external_tasks = [
                    self._collect_robots_txt(page.url),
                    self._collect_sitemap_analysis(page.url)
                ]
                robots_result, sitemap_result = await asyncio.gather(*external_tasks)
                result["robots_txt"] = robots_result
                result["sitemap"] = sitemap_result
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            
            # Generate summary
            result["summary"] = self._generate_combined_summary(result)
            
            logger.info(f"Optimized analysis completed in {processing_time}s")
            return result
            
        except Exception as e:
            logger.error(f"Analysis error for {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'processing_time': time.time() - start_time
            }
        finally:
            # Cleanup
            if context:
                try:
                    if not context.is_closed:
                        await context.close()
                except:
                    pass
            if browser_info:
                await browser_manager.release_context(context, browser_info)
    
    async def _navigate_with_retry(self, page, url):
        """Optimized navigation with shorter timeouts"""
        try:
            # First attempt: quick navigation
            return await page.goto(url, wait_until='domcontentloaded', timeout=20000)
        except:
            try:
                # Second attempt: basic navigation
                return await page.goto(url, timeout=15000)
            except:
                # Final attempt: minimal timeout
                return await page.goto(url, timeout=10000)
    
    # ==========================================
    # OPTIMIZED ANALYSIS METHODS
    # ==========================================
    
    async def _analyze_performance(self, page, soup, response) -> Dict[str, Any]:
        """Optimized performance analysis"""
        
        # Quick performance timing
        try:
            performance_timing = await page.evaluate("""
                () => {
                    const timing = performance.timing;
                    const paint = performance.getEntriesByType('paint');
                    return {
                        navigationStart: timing.navigationStart,
                        loadEventEnd: timing.loadEventEnd,
                        domContentLoadedEventEnd: timing.domContentLoadedEventEnd,
                        responseStart: timing.responseStart,
                        firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || null,
                        firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || null
                    };
                }
            """)
        except:
            performance_timing = {}
        
        # Calculate timing metrics
        ttfb_ms = performance_timing.get('responseStart', 0) - performance_timing.get('navigationStart', 0) if performance_timing else 1000
        
        # Quick Core Web Vitals estimation
        core_web_vitals = {
            "LCP_ms": await self._estimate_lcp(soup),
            "CLS": 0.0,
            "INP_ms": None,
            "TTFB_ms": max(ttfb_ms, 200),
            "TTI_ms": await self._estimate_tti(soup, ttfb_ms),
            "FCP_ms": performance_timing.get('firstContentfulPaint') or (ttfb_ms + 200),
            "speed_index": await self._estimate_speed_index(soup),
            "source": "lab_estimate"
        }
        
        # Quick CDN detection
        headers = dict(response.headers) if response else {}
        cdn_provider = self._detect_cdn(headers)
        
        # Resource analysis
        resources = await self._analyze_resources(soup)
        
        # Page size
        html_content = str(soup)
        page_size_bytes = len(html_content.encode('utf-8'))
        
        return {
            "http_version": headers.get('server', 'unknown'),
            "server_geo": None,
            "cdn_provider": cdn_provider,
            "headers": {
                "cache_control": headers.get('cache-control'),
                "etag": headers.get('etag'),
                "last_modified": headers.get('last-modified'),
                "content_encoding": headers.get('content-encoding')
            },
            "core_web_vitals": core_web_vitals,
            "page_size": {
                "bytes": page_size_bytes,
                "kb": round(page_size_bytes / 1024, 2),
                "mb": round(page_size_bytes / 1024 / 1024, 2)
            },
            "resource_analysis": resources,
            "performance_score": self._calculate_performance_score(core_web_vitals, resources)
        }
    
    async def _estimate_lcp(self, soup) -> float:
        """Quick LCP estimation"""
        large_elements = soup.find_all(['img', 'video', 'h1', 'p'], limit=5)  # Reduced limit
        base_lcp = 1500
        if len(large_elements) > 3:
            base_lcp += 300
        return base_lcp
    
    async def _estimate_tti(self, soup, ttfb: float) -> float:
        """Quick TTI estimation"""
        script_tags = soup.find_all('script', limit=10)  # Limit for speed
        js_complexity = len(script_tags) * 50  # Reduced complexity
        return ttfb + js_complexity + 500
    
    async def _estimate_speed_index(self, soup) -> float:
        """Quick Speed Index estimation"""
        elements = len(soup.find_all(limit=100))  # Limit for speed
        return 1000 + (elements * 1)
    
    def _detect_cdn(self, headers: Dict[str, str]) -> Optional[str]:
        """Quick CDN detection"""
        cdn_indicators = {
            'cloudflare': ['cf-ray', 'server'],
            'fastly': ['fastly-io'],
            'aws': ['x-amz'],
            'maxcdn': ['x-maxcdn'],
            'akamai': ['x-akamai']
        }
        
        headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
        
        for cdn, indicators in cdn_indicators.items():
            for indicator in indicators:
                if indicator in headers_lower:
                    if cdn == 'cloudflare' and 'cloudflare' in headers_lower.get('server', ''):
                        return 'Cloudflare'
        return None
    
    async def _analyze_resources(self, soup) -> Dict[str, Any]:
        """Quick resource analysis"""
        css_links = soup.find_all('link', {'rel': 'stylesheet'}, limit=20)
        js_scripts = soup.find_all('script', {'src': True}, limit=20)
        images = soup.find_all('img', limit=30)
        
        return {
            "css_resources": [{'url': link.get('href'), 'is_external': link.get('href', '').startswith('http')} for link in css_links[:10]],
            "js_resources": [{'url': script.get('src'), 'is_external': script.get('src', '').startswith('http')} for script in js_scripts[:10]],
            "image_resources": [{'url': img.get('src'), 'alt': img.get('alt', '')} for img in images[:15]],
            "external_css_count": len([link for link in css_links if link.get('href', '').startswith('http')]),
            "external_js_count": len([script for script in js_scripts if script.get('src', '').startswith('http')]),
            "total_resources": len(css_links) + len(js_scripts) + len(images)
        }
    
    def _calculate_performance_score(self, vitals: Dict, resources: Dict) -> int:
        """Quick performance score calculation"""
        score = 100
        if vitals.get('TTFB_ms', 0) > 600:
            score -= 20
        if vitals.get('LCP_ms', 0) > 2500:
            score -= 25
        if vitals.get('TTI_ms', 0) > 3800:
            score -= 15
        if resources.get('total_resources', 0) > 50:
            score -= 10
        return max(score, 0)
    
    async def _analyze_seo_structured(self, soup, url, response) -> Dict[str, Any]:
        """Optimized SEO analysis"""
        title_tag = soup.find('title')
        title = title_tag.text.strip() if title_tag else ""
        
        meta_desc = soup.find('meta', {'name': 'description'})
        meta_description = meta_desc.get('content', '') if meta_desc else ""
        
        canonical = soup.find('link', {'rel': 'canonical'})
        canonical_url = canonical.get('href') if canonical else None
        
        robots_meta = soup.find('meta', {'name': 'robots'})
        robots_content = robots_meta.get('content', '') if robots_meta else ""
        
        robots_directives = {
            'noindex': 'noindex' in robots_content.lower(),
            'nofollow': 'nofollow' in robots_content.lower(),
            'is_indexable': 'noindex' not in robots_content.lower(),
            'is_followable': 'nofollow' not in robots_content.lower()
        }
        
        # Quick heading analysis
        h1_tags = soup.find_all('h1')
        h2_tags = soup.find_all('h2')
        
        heading_structure = {
            'h1_count': len(h1_tags),
            'h2_count': len(h2_tags),
            'proper_h1_usage': len(h1_tags) == 1,
            'headings_by_level': {
                'h1': [{'text': h.get_text().strip()[:100]} for h in h1_tags[:3]],
                'h2': [{'text': h.get_text().strip()[:100]} for h in h2_tags[:5]]
            }
        }
        
        # Quick word count
        text_content = soup.get_text()
        word_count = len(text_content.split())
        reading_time = max(1, word_count // 200)
        
        seo_score = self._calculate_seo_score(len(title), meta_description, heading_structure, word_count)
        
        return {
            "title_analysis": {
                "title": title,
                "length": len(title),
                "word_count": len(title.split()),
                "is_optimal_length": 30 <= len(title) <= 60
            },
            "meta_description": {
                "description": meta_description,
                "length": len(meta_description),
                "is_optimal_length": 120 <= len(meta_description) <= 160,
                "exists": bool(meta_description)
            },
            "robots_meta": robots_directives,
            "canonical_url": {
                "exists": bool(canonical_url),
                "url": canonical_url
            },
            "heading_structure": heading_structure,
            "content_metrics": {
                "word_count": word_count,
                "reading_time_minutes": reading_time,
                "is_sufficient_content": word_count >= 300
            },
            "seo_score": seo_score
        }
    
    def _calculate_seo_score(self, title_length: int, meta_desc: str, headings: Dict, word_count: int) -> int:
        """Quick SEO score calculation"""
        score = 100
        if not (30 <= title_length <= 60):
            score -= 15
        if not meta_desc:
            score -= 20
        elif not (120 <= len(meta_desc) <= 160):
            score -= 10
        if headings['h1_count'] != 1:
            score -= 15
        if headings['h2_count'] == 0:
            score -= 10
        if word_count < 300:
            score -= 20
        return max(score, 0)
    
    async def _analyze_security(self, url: str, response) -> Dict[str, Any]:
        """Quick security analysis"""
        headers = dict(response.headers) if response else {}
        
        security_headers = {
            "strict_transport_security": bool(headers.get('strict-transport-security')),
            "content_security_policy": bool(headers.get('content-security-policy')),
            "x_frame_options": bool(headers.get('x-frame-options')),
            "x_content_type_options": bool(headers.get('x-content-type-options')),
            "referrer_policy": bool(headers.get('referrer-policy')),
            "permissions_policy": bool(headers.get('permissions-policy'))
        }
        
        is_https = url.startswith('https://')
        security_score = sum(security_headers.values()) * 15
        if is_https:
            security_score += 10
        
        return {
            "https_usage": is_https,
            "security_headers": security_headers,
            "missing_headers": [k for k, v in security_headers.items() if not v],
            "security_score": min(security_score, 100),
            "recommendations": self._get_security_recommendations(security_headers, is_https)
        }
    
    def _get_security_recommendations(self, headers: Dict[str, bool], is_https: bool) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        if not is_https:
            recommendations.append("Enable HTTPS/SSL")
        for header, present in headers.items():
            if not present:
                header_name = header.replace('_', '-').upper()
                recommendations.append(f"Add {header_name} security header")
        return recommendations
    
    async def _analyze_content_structured(self, soup) -> Dict[str, Any]:
        """Quick content analysis"""
        text_content = soup.get_text()
        words = text_content.split()
        word_count = len(words)
        character_count = len(text_content)
        reading_time = max(1, word_count // 200)
        
        # Quick paragraph analysis
        paragraphs = soup.find_all('p', limit=10)
        paragraph_data = []
        for p in paragraphs:
            p_text = p.get_text().strip()
            if p_text:
                paragraph_data.append({
                    'text': p_text[:50] + '...' if len(p_text) > 50 else p_text,
                    'word_count': len(p_text.split()),
                    'has_links': bool(p.find_all('a'))
                })
        
        # Quick list analysis
        lists = soup.find_all(['ul', 'ol'], limit=5)
        list_data = []
        for lst in lists:
            items = lst.find_all('li')
            list_data.append({
                'type': lst.name,
                'item_count': len(items),
                'items': [item.get_text().strip()[:30] for item in items[:3]]
            })
        
        return {
            "text_content": text_content[:300] + '...' if len(text_content) > 300 else text_content,
            "word_count": word_count,
            "character_count": character_count,
            "reading_time": reading_time,
            "paragraphs": {
                "total_paragraphs": len(paragraphs),
                "paragraphs": paragraph_data,
                "average_length": sum(len(p.get_text().split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
            },
            "lists": {
                "total_lists": len(lists),
                "lists": list_data,
                "total_list_items": sum(len(lst.find_all('li')) for lst in lists)
            },
            "content_density": round(word_count / len(text_content) if text_content else 0, 3)
        }
    
    async def _analyze_images_structured(self, soup, base_url) -> Dict[str, Any]:
        """Quick image analysis"""
        images = soup.find_all('img', limit=20)  # Limit for speed
        
        image_data = []
        images_with_alt = 0
        lazy_loaded = 0
        format_distribution = {}
        
        for img in images:
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            if alt:
                images_with_alt += 1
            
            loading = img.get('loading', '')
            if loading == 'lazy':
                lazy_loaded += 1
            
            if src:
                ext = src.split('.')[-1].lower().split('?')[0]
                format_distribution[ext] = format_distribution.get(ext, 0) + 1
            
            image_data.append({
                'src': src,
                'alt': alt[:50],  # Truncate for speed
                'has_alt': bool(alt),
                'loading': loading,
                'format': ext if src else 'unknown',
                'is_lazy_loaded': loading == 'lazy'
            })
        
        return {
            "total_images": len(images),
            "images": image_data,
            "images_with_alt": images_with_alt,
            "images_without_alt": len(images) - images_with_alt,
            "lazy_loaded_images": lazy_loaded,
            "format_distribution": format_distribution,
            "alt_text_quality": {
                "descriptive_alt": len([img for img in image_data if len(img['alt']) > 10]),
                "empty_alt": len([img for img in image_data if img['alt'] == '']),
                "missing_alt": len([img for img in image_data if not img['has_alt']])
            }
        }
    
    async def _analyze_technical_structured(self, page, soup, response) -> Dict[str, Any]:
        """Quick technical analysis"""
        headers = dict(response.headers) if response else {}
        
        html_validation = {
            "doctype": "html5" if '<!DOCTYPE html>' in str(soup)[:200] else "unknown",
            "lang_attribute": bool(soup.find('html', {'lang': True})),
            "charset_declared": bool(soup.find('meta', {'charset': True}))
        }
        
        # Quick resource count
        external_stylesheets = len(soup.find_all('link', {'rel': 'stylesheet', 'href': lambda x: x and x.startswith('http')}, limit=10))
        external_scripts = len(soup.find_all('script', {'src': lambda x: x and x.startswith('http')}, limit=10))
        
        html_content = str(soup)
        content_length = len(html_content.encode('utf-8'))
        
        return {
            "html_size": {
                "bytes": content_length,
                "kb": round(content_length / 1024, 2),
                "mb": round(content_length / 1024 / 1024, 2)
            },
            "response_headers": dict(list(headers.items())[:10]),  # Limit headers
            "html_validation": html_validation,
            "resource_analysis": {
                "external_stylesheets": external_stylesheets,
                "external_scripts": external_scripts
            },
            "encoding": headers.get('content-encoding')
        }
    
    async def _analyze_links_structured(self, soup, base_url) -> Dict[str, Any]:
        """Quick links analysis"""
        links = soup.find_all('a', href=True, limit=30)  # Limit for speed
        
        internal_links = []
        external_links = []
        email_links = []
        phone_links = []
        
        parsed_base = urlparse(base_url)
        
        for link in links:
            href = link.get('href', '')
            text = link.get_text().strip()[:50]  # Truncate
            
            if href.startswith('mailto:'):
                email_links.append({'url': href, 'text': text})
            elif href.startswith('tel:'):
                phone_links.append({'url': href, 'text': text})
            elif href.startswith('http'):
                parsed_href = urlparse(href)
                if parsed_href.netloc == parsed_base.netloc:
                    internal_links.append({'url': href, 'text': text})
                else:
                    external_links.append({'url': href, 'text': text})
            else:
                internal_links.append({'url': urljoin(base_url, href), 'text': text})
        
        return {
            "total_links": len(links),
            "internal_links": {"count": len(internal_links), "links": internal_links[:10]},
            "external_links": {"count": len(external_links), "links": external_links[:10]},
            "email_links": {"count": len(email_links), "links": email_links},
            "phone_links": {"count": len(phone_links), "links": phone_links}
        }
    
    async def _analyze_structured_data(self, soup) -> Dict[str, Any]:
        """Quick structured data analysis"""
        json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'}, limit=5)
        json_ld_data = []
        schema_types = set()
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                json_ld_data.append(data)
                if isinstance(data, dict) and '@type' in data:
                    schema_types.add(data['@type'])
            except:
                continue
        
        # Quick OpenGraph and Twitter analysis
        og_tags = {}
        twitter_tags = {}
        
        for meta in soup.find_all('meta', limit=50):
            property_attr = meta.get('property', '')
            name_attr = meta.get('name', '')
            
            if property_attr.startswith('og:'):
                og_tags[property_attr] = meta.get('content', '')
            elif name_attr.startswith('twitter:'):
                twitter_tags[name_attr] = meta.get('content', '')
        
        return {
            "json_ld": json_ld_data,
            "opengraph": og_tags,
            "twitter_cards": twitter_tags,
            "schema_types": list(schema_types),
            "summary": {
                "has_structured_data": bool(json_ld_data),
                "total_json_ld": len(json_ld_data),
                "total_schema_types": len(schema_types),
                "has_social_meta": bool(og_tags or twitter_tags)
            }
        }
    
    async def _analyze_mobile(self, soup) -> Dict[str, Any]:
        """Quick mobile analysis"""
        viewport_meta = soup.find('meta', {'name': 'viewport'})
        viewport_content = viewport_meta.get('content', '') if viewport_meta else ''
        
        apple_touch_icon = soup.find('link', {'rel': lambda x: x and 'apple-touch-icon' in x})
        
        return {
            "viewport_meta": {
                "exists": bool(viewport_meta),
                "content": viewport_content,
                "is_responsive": "width=device-width" in viewport_content
            },
            "mobile_specific_elements": {
                "apple_touch_icon": bool(apple_touch_icon)
            }
        }
    
    async def _analyze_accessibility(self, soup) -> Dict[str, Any]:
        """Quick accessibility analysis"""
        images = soup.find_all('img', limit=20)
        images_with_alt = len([img for img in images if img.get('alt')])
        
        h1_tags = soup.find_all('h1')
        html_tag = soup.find('html')
        has_lang = bool(html_tag and html_tag.get('lang'))
        
        return {
            "images": {
                "total_images": len(images),
                "images_with_alt": images_with_alt,
                "images_without_alt": len(images) - images_with_alt
            },
            "headings": {
                "h1_count": len(h1_tags),
                "proper_h1_usage": len(h1_tags) == 1
            },
            "language_declaration": {
                "html_has_lang": has_lang
            }
        }
    
    # ==========================================
    # LIGHTWEIGHT DETAILED ANALYSIS METHODS
    # ==========================================
    
    def _collect_page_info(self, url, soup) -> Dict[str, Any]:
        """Quick page information"""
        parsed_url = urlparse(url)
        title = soup.find('title')
        
        return {
            'url': url,
            'protocol': parsed_url.scheme,
            'domain': parsed_url.netloc.replace('www.', ''),
            'title': title.text.strip()[:200] if title else '',
            'language': soup.find('html', lang=True).get('lang', 'unknown') if soup.find('html', lang=True) else 'unknown',
            'charset': self._extract_charset(soup)
        }
    
    def _extract_charset(self, soup):
        """Extract character encoding quickly"""
        charset_meta = soup.find('meta', charset=True)
        if charset_meta:
            return charset_meta.get('charset', 'UTF-8')
        return 'UTF-8'
    
    async def _collect_business_info(self, soup, url) -> Dict[str, Any]:
        """Quick business information extraction"""
        business_info = {
            'company_name': '',
            'addresses': [],
            'email_addresses': [],
            'phone_numbers': []
        }
        
        title = soup.find('title')
        if title:
            business_info['company_name'] = title.text.strip()[:100]
        
        text_content = soup.get_text()[:5000]  # Limit text for speed
        
        # Quick email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text_content)
        business_info['email_addresses'] = list(set(emails))[:5]
        
        # Quick Dutch phone extraction
        phone_pattern = r'\+31[\s\-]?\d{1,3}[\s\-]?\d{3}[\s\-]?\d{4}|0\d{1,3}[\s\-]?\d{3}[\s\-]?\d{4}'
        phones = re.findall(phone_pattern, text_content)
        business_info['phone_numbers'] = list(set(phones))[:3]
        
        return business_info
    
    async def _collect_contact_info(self, soup) -> Dict[str, Any]:
        """Quick contact information"""
        contact_pages = []
        contact_keywords = ['contact', 'over', 'about']
        
        for link in soup.find_all('a', href=True, limit=20):
            href = link.get('href', '').lower()
            text = link.get_text(strip=True).lower()
            
            if any(keyword in href or keyword in text for keyword in contact_keywords):
                contact_pages.append({
                    'url': link.get('href'),
                    'text': link.get_text(strip=True)[:50]
                })
        
        return {'contact_pages': contact_pages[:5]}
    
    async def _collect_enhanced_content(self, soup) -> Dict[str, Any]:
        """Quick enhanced content analysis"""
        text_content = soup.get_text()
        clean_text = re.sub(r'\s+', ' ', text_content).strip()
        words = clean_text.split()
        
        # Quick headings
        headings = {}
        for i in range(1, 4):  # Only H1-H3 for speed
            h_tags = soup.find_all(f'h{i}', limit=5)
            if h_tags:
                headings[f'h{i}'] = [{'text': h.get_text(strip=True)[:100]} for h in h_tags]
        
        return {
            'word_count': len(words),
            'reading_time': max(1, len(words) // 200),
            'text_content': clean_text[:1000] + '...' if len(clean_text) > 1000 else clean_text,
            'headings': headings,
            'language': 'nl-NL'
        }
    
    async def _collect_enhanced_links(self, soup, base_url) -> Dict[str, Any]:
        """Quick enhanced links analysis"""
        links = {'all': [], 'internal': [], 'external': []}
        
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True, limit=25):
            href = link.get('href', '')
            text = link.get_text(strip=True)[:50]
            
            link_info = {'url': href, 'text': text}
            links['all'].append(link_info)
            
            if href.startswith('http'):
                link_domain = urlparse(href).netloc
                if link_domain == base_domain:
                    links['internal'].append(link_info)
                else:
                    links['external'].append(link_info)
            else:
                links['internal'].append(link_info)
        
        return {
            'all': links['all'][:20],
            'internal': links['internal'][:15],
            'external': links['external'][:10],
            'internal_links_count': len(links['internal']),
            'external_links_count': len(links['external'])
        }
    
    async def _collect_enhanced_images(self, soup, base_url) -> List[Dict[str, Any]]:
        """Quick enhanced images analysis"""
        images = []
        
        for img in soup.find_all('img', limit=15):
            src = img.get('src', '')
            if src:
                absolute_url = urljoin(base_url, src) if not src.startswith('http') else src
                
                images.append({
                    'src': absolute_url,
                    'alt': img.get('alt', '')[:100],
                    'loading': img.get('loading', ''),
                    'format': src.split('.')[-1].lower() if '.' in src else 'unknown'
                })
        
        return images
    
    async def _collect_meta_data(self, soup) -> Dict[str, Any]:
        """Quick meta data collection"""
        meta_data = {}
        
        for meta in soup.find_all('meta', limit=30):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            
            if name and content:
                meta_data[name.lower()] = content[:200]  # Truncate
        
        return meta_data
    
    async def _collect_page_structure(self, soup) -> Dict[str, Any]:
        """Quick page structure analysis"""
        semantic_tags = ['nav', 'header', 'main', 'section', 'article', 'footer']
        semantic_elements = []
        
        for tag in semantic_tags:
            count = len(soup.find_all(tag, limit=10))
            if count > 0:
                semantic_elements.append({'tag': tag, 'count': count})
        
        return {
            'semantic_elements': semantic_elements,
            'has_nav': bool(soup.find('nav')),
            'has_footer': bool(soup.find('footer'))
        }
    
    async def _collect_enhanced_seo(self, soup, url) -> Dict[str, Any]:
        """Quick enhanced SEO"""
        title = soup.find('title')
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        
        h1_count = len(soup.find_all('h1'))
        h1_text = [h1.get_text(strip=True)[:100] for h1 in soup.find_all('h1', limit=3)]
        
        text_content = soup.get_text()
        word_count = len(text_content.split())
        
        return {
            'title_text': title.text.strip()[:200] if title else '',
            'title_length': len(title.text) if title else 0,
            'meta_description_text': meta_desc.get('content', '')[:200] if meta_desc else '',
            'meta_description_length': len(meta_desc.get('content', '')) if meta_desc else 0,
            'h1_count': h1_count,
            'h1_text': h1_text,
            'word_count_estimate': word_count
        }
    
    async def _collect_social_media(self, soup) -> Dict[str, Any]:
        """Quick social media collection"""
        open_graph = {}
        twitter_cards = {}
        
        for meta in soup.find_all('meta', limit=30):
            property_attr = meta.get('property', '')
            name_attr = meta.get('name', '')
            
            if property_attr.startswith('og:'):
                open_graph[property_attr.replace('og:', '')] = meta.get('content', '')[:200]
            elif name_attr.startswith('twitter:'):
                twitter_cards[name_attr.replace('twitter:', '')] = meta.get('content', '')[:200]
        
        return {
            'open_graph': open_graph,
            'twitter_cards': twitter_cards,
            'summary': {
                'has_open_graph': bool(open_graph),
                'has_twitter_cards': bool(twitter_cards)
            }
        }
    
    async def _collect_technical_detailed(self, page, soup, response) -> Dict[str, Any]:
        """Quick technical details"""
        response_headers = {}
        if response:
            response_headers = dict(list(response.headers.items())[:10])  # Limit
        
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        
        return {
            'response_headers': response_headers,
            'mobile_friendly': {
                'has_viewport': bool(viewport_meta),
                'viewport_content': viewport_meta.get('content', '') if viewport_meta else ''
            },
            'html_size': len(str(soup)),
            'security': {
                'https': page.url.startswith('https://')
            }
        }
    
    async def _collect_robots_txt(self, url) -> Dict[str, Any]:
        """Quick robots.txt check"""
        try:
            robots_url = urljoin(url, '/robots.txt')
            
            timeout = aiohttp.ClientTimeout(total=5)  # Reduced timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            'url': robots_url,
                            'status': response.status,
                            'content': content[:1000],  # Truncate
                            'size': len(content)
                        }
                    else:
                        return {'url': robots_url, 'status': response.status, 'content': None}
        except:
            return {'url': urljoin(url, '/robots.txt'), 'status': 0, 'content': None}
    
    async def _collect_sitemap_analysis(self, url) -> List[Dict[str, Any]]:
        """Quick sitemap check"""
        sitemap_url = urljoin(url, '/sitemap.xml')
        
        try:
            timeout = aiohttp.ClientTimeout(total=5)  # Reduced timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return [{
                            'url': sitemap_url,
                            'status': response.status,
                            'size': len(content),
                            'url_count': content.count('<url>')
                        }]
        except:
            pass
        
        return []
    
    def _generate_combined_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Quick combined summary generation"""
        scores = {}
        
        # Extract scores
        if 'performance' in analysis_results:
            scores['performance_score'] = analysis_results['performance'].get('performance_score', 0)
        
        if 'seo_analysis' in analysis_results:
            scores['seo_score'] = analysis_results['seo_analysis'].get('seo_score', 0)
        
        if 'security_analysis' in analysis_results:
            scores['security_score'] = analysis_results['security_analysis'].get('security_score', 0)
        
        # Quick overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0
        
        # Quick recommendations
        recommendations = []
        
        if scores.get('performance_score', 0) < 80:
            recommendations.append("Improve page loading speed")
        if scores.get('seo_score', 0) < 80:
            recommendations.append("Optimize SEO elements")
        if scores.get('security_score', 0) < 80:
            recommendations.append("Add security headers")
        
        return {
            "overall_scores": {
                **scores,
                "overall_score": round(overall_score, 1)
            },
            "recommendations": recommendations,
            "analysis_summary": {
                "pages_analyzed": 1,
                "total_issues": len(recommendations)
            }
        }

# =====================
# API ENDPOINTS (unchanged)
# =====================

combined_analyzer = OptimizedCombinedWebsiteAnalyzer()

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "optimization": "HIGH_PERFORMANCE",
        "features": "Optimized for sequential scans"
    }

@app.post("/analyze/combined")
async def analyze_website_combined(request: CombinedAnalyzeRequest):
    """
    OPTIMIZED Combined website analysis for high-performance sequential scans
    """
    try:
        results = await combined_analyzer.analyze_website(str(request.url), request)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in combined analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analyze/combined/{url:path}")
async def analyze_website_combined_get(url: str):
    """
    OPTIMIZED Quick combined analysis via GET request
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError("Invalid URL format")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    request = CombinedAnalyzeRequest(url=url)
    
    try:
        results = await combined_analyzer.analyze_website(url, request)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in quick combined analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Add Core Web Vitals endpoint for Google API testing
@app.get("/vitals/{url:path}")
async def get_core_web_vitals(url: str, device: str = "mobile"):
    """
    Get real Core Web Vitals data from Google APIs (for testing)
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        vitals = await get_real_core_web_vitals(url, device)
        return vitals
    except Exception as e:
        logger.error(f"Vitals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Combined Ultimate Website Analyzer API - PERFORMANCE OPTIMIZED",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "status": "operational",
        "optimization_level": "HIGH_PERFORMANCE",
        "description": "Optimized for handling 5+ sequential scans for competitive analysis",
        "performance_features": [
            "🚀 Browser pooling for faster initialization",
            "⚡ Reduced timeouts for quicker responses",
            "🔄 Optimized resource management",
            "💾 Smart caching with shorter TTL",
            "🎯 Parallel processing where possible",
            "🗑️ Automatic cleanup and garbage collection",
            "⏱️ Progressive timeout strategy",
            "🔧 Lightweight analysis methods"
        ],
        "endpoints": {
            "POST /analyze/combined": "Full optimized analysis",
            "GET /analyze/combined/{url}": "Quick optimized analysis",
            "GET /vitals/{url}": "Real Core Web Vitals (Google APIs)"
        },
        "performance_stats": {
            "target_scan_time": "15-30 seconds per URL",
            "sequential_capacity": "5+ scans",
            "timeout_limit": f"{TIMEOUT_SECONDS} seconds",
            "browser_pool_size": "3 browsers"
        }
    }

# =====================
# STARTUP & CLEANUP (optimized)
# =====================

@app.on_event("startup")
async def startup_event():
    """Initialize optimized browser pool on startup"""
    logger.info("OPTIMIZED Combined Website Analyzer API starting up...")
    try:
        await browser_manager.initialize()
        logger.info("Optimized browser pool initialized successfully")
    except Exception as e:
        logger.error(f"Browser pool initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down OPTIMIZED analyzer...")
    await browser_manager.close()
    gc.collect()  # Force garbage collection
    logger.info("Cleanup completed")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    if ENVIRONMENT == "production":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=MAX_WORKERS,
            access_log=False,
            log_level="info"
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="debug"
        )
