"""
Combined Ultimate Website Analyzer API v4.1
Combines structured + detailed analysis + REAL Google Core Web Vitals
WITH FIXED BROWSER MANAGEMENT FOR SEQUENTIAL SCANS
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
import gc  # Added for memory cleanup
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = os.getenv("API_VERSION", "4.1.0")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "300"))

# Google API Keys for real Core Web Vitals
PSI_KEY = os.getenv("PSI_API_KEY")  # PageSpeed Insights API Key
CRUX_KEY = os.getenv("CRUX_API_KEY")  # Chrome UX Report API Key

# Google API URLs
PSI_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
CRUX_URL = "https://chromeuxreport.googleapis.com/v1/records:query"

# Cache for Google API results (12 hours TTL)
VITALS_CACHE = TTLCache(maxsize=1000, ttl=60*60*12)

app = FastAPI(
    title="Combined Ultimate Website Analyzer API",
    description="Complete website analysis with both structured metrics AND detailed content analysis + REAL Google Core Web Vitals",
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

# Add request logging middleware
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
# MODELS
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
    include_real_vitals: bool = True  # NEW: Real Google Core Web Vitals
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
    use_origin_fallback: bool = True  # NEW: For CrUX data fallback

# =====================
# GOOGLE API UTILITIES
# =====================

def ms_to_s(v: Optional[float]) -> Optional[float]:
    """Convert milliseconds to seconds with 3 decimal precision"""
    return round(v/1000.0, 3) if isinstance(v, (int, float)) else None

def get_rating(metric: str, value: Optional[float]) -> Optional[str]:
    """Get rating label for Core Web Vitals metrics"""
    if value is None: 
        return None
    
    thresholds = {
        "LCP": (2.5, 4.0),      # good <= 2.5s, poor > 4.0s
        "INP": (0.2, 0.5),      # good <= 200ms, poor > 500ms
        "CLS": (0.1, 0.25),     # good <= 0.1, poor > 0.25
        "FID": (0.1, 0.3)       # good <= 100ms, poor > 300ms
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
    """
    Get real CrUX field data for the given URL
    """
    if not CRUX_KEY:
        return {"available": False, "reason": "CRUX_API_KEY not configured"}

    # First attempt: exact URL (page-level data)
    payload = {
        "url": url,
        "formFactor": form_factor
    }
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30) as client:
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
                # Extract origin from URL
                parsed = urlparse(url)
                origin = f"{parsed.scheme}://{parsed.netloc}"
                
                origin_payload = {
                    "origin": origin,
                    "formFactor": form_factor
                }
                
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
        "FID_s": get_p75("first_input_delay", True)  # Legacy metric
    }

async def get_psi_data(url: str, strategy: str = "mobile") -> Dict[str, Any]:
    """
    Get real PageSpeed Insights (Lighthouse) lab data
    """
    if not PSI_KEY:
        return {"available": False, "reason": "PSI_API_KEY not configured"}

    params = {
        "url": url,
        "strategy": strategy,
        "category": "performance",
        "key": PSI_KEY
    }

    async with httpx.AsyncClient(timeout=90) as client:
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
            
            # Performance score
            perf_score = lighthouse_result.get("categories", {}).get("performance", {}).get("score")
            if isinstance(perf_score, (int, float)):
                perf_score = round(perf_score * 100)

            # Core Web Vitals
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
    """
    Combine field data (CrUX) and lab data (PSI) with field data priority
    """
    # Use field data as primary, lab data as fallback
    lcp = field_data.get("LCP_s") if field_data.get("LCP_s") is not None else lab_data.get("LCP_s")
    inp = field_data.get("INP_s") if field_data.get("INP_s") is not None else lab_data.get("INP_s")
    cls = field_data.get("CLS") if field_data.get("CLS") is not None else lab_data.get("CLS")
    
    return {
        "LCP_s": lcp,
        "INP_s": inp,
        "CLS": cls,
        "FID_s": field_data.get("FID_s"),  # Only from CrUX
        "ratings": {
            "LCP": get_rating("LCP", lcp),
            "INP": get_rating("INP", inp),
            "CLS": get_rating("CLS", cls)
        }
    }

async def get_real_core_web_vitals(url: str, device: str = "mobile", use_origin_fallback: bool = True) -> Dict[str, Any]:
    """
    Get REAL Core Web Vitals from Google APIs
    """
    # Cache check
    cache_key = (url, device)
    if cache_key in VITALS_CACHE:
        cached_result = VITALS_CACHE[cache_key].copy()
        cached_result["from_cache"] = True
        return cached_result

    # Determine parameters
    form_factor = "PHONE" if device.lower() == "mobile" else "DESKTOP"
    strategy = "mobile" if device.lower() == "mobile" else "desktop"
    
    # Get data (parallel)
    field_data, lab_data = await asyncio.gather(
        get_crux_data(url, form_factor, use_origin_fallback),
        get_psi_data(url, strategy)
    )
    
    # Combine results
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
# FIXED BROWSER MANAGER
# =====================

class FixedBrowserManager:
    """Fixed browser manager - creates fresh browser every 2 requests"""
    
    def __init__(self):
        self._browser = None
        self._playwright = None
        self._request_count = 0
        self._lock = asyncio.Lock()
    
    async def get_context(self, viewport=None, user_agent=None):
        """Get browser context with fresh browser every 2 requests"""
        async with self._lock:
            self._request_count += 1
            
            # Create fresh browser every 2 requests OR if browser is None
            if self._browser is None or self._request_count % 2 == 0:
                await self._recreate_browser()
            
            try:
                context = await self._browser.new_context(
                    viewport=viewport or {'width': 1366, 'height': 768},
                    user_agent=user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Cache-Control': 'no-cache',
                        'Pragma': 'no-cache'
                    }
                )
                return context
            except Exception as e:
                logger.warning(f"Context creation failed, recreating browser: {e}")
                await self._recreate_browser()
                context = await self._browser.new_context(
                    viewport=viewport or {'width': 1366, 'height': 768},
                    user_agent=user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                return context
    
    async def _recreate_browser(self):
        """Recreate browser completely"""
        try:
            # Close existing browser
            if self._browser:
                await self._browser.close()
                await asyncio.sleep(0.5)  # Give it time to close
            
            # Close existing playwright
            if self._playwright:
                await self._playwright.stop()
                await asyncio.sleep(0.5)
            
            # Force garbage collection
            gc.collect()
            
            # Create fresh playwright and browser
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-zygote',
                    '--disable-blink-features=AutomationControlled',
                    '--memory-pressure-off',
                    '--max-old-space-size=256',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                ]
            )
            logger.info(f"Fresh browser created for request #{self._request_count}")
            
        except Exception as e:
            logger.error(f"Browser recreation failed: {e}")
            raise
    
    async def close_context(self, context):
        """Close context and cleanup"""
        try:
            if context and hasattr(context, 'close'):
                await context.close()
                await asyncio.sleep(0.2)  # Give context time to close
            
            # Memory cleanup every request
            gc.collect()
                
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
    
    async def close(self):
        """Close browser and playwright"""
        try:
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            logger.info("Browser manager closed")
        except Exception as e:
            logger.warning(f"Error closing browser manager: {e}")

# Global browser manager
browser_manager = FixedBrowserManager()

# =====================
# COMBINED ANALYZER SERVICE
# =====================

class CombinedWebsiteAnalyzer:
    """Combined analyzer that provides both structured metrics AND detailed analysis"""
    
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    async def analyze_website(self, url: str, options: CombinedAnalyzeRequest) -> Dict[str, Any]:
        """Main analysis function that combines both analysis types"""
        start_time = time.time()
        
        context = None
        page = None
        
        try:
            logger.info(f"Starting combined analysis for: {url}")
            
            # Get browser context
            viewport = {'width': 1920 if options.device == Device.desktop else 390, 
                       'height': 1080 if options.device == Device.desktop else 844}
            context = await browser_manager.get_context(viewport=viewport)
            page = await context.new_page()
            
            # Navigate to page
            response = await self._navigate_with_retry(page, url)
            await page.wait_for_timeout(3000)
            
            # Get HTML content
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # ==========================================
            # COMBINED RESULT STRUCTURE
            # ==========================================
            
            result = {
                # Basic info from both systems
                "url": url,
                "final_url": page.url,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": 0,  # Updated at end
                "status_code": response.status if response else None,
                "api_version": API_VERSION
            }
            
            # ==========================================
            # PART 1: STRUCTURED ANALYSIS (Original API)
            # ==========================================
            
            # Performance Analysis
            if options.include_performance:
                logger.info("Running performance analysis...")
                result["performance"] = await self._analyze_performance(page, soup, response)
            
            # SEO Analysis (Structured)
            if options.include_seo:
                logger.info("Running SEO analysis...")
                result["seo_analysis"] = await self._analyze_seo_structured(soup, url, response)
            
            # Security Analysis
            if options.include_security:
                logger.info("Running security analysis...")
                result["security_analysis"] = await self._analyze_security(page.url, response)
            
            # Content Analysis (Structured)
            result["content_analysis"] = await self._analyze_content_structured(soup)
            
            # Images Analysis (Structured)
            if options.include_images:
                result["images_analysis"] = await self._analyze_images_structured(soup, page.url)
            
            # Technical Analysis (Structured)
            if options.include_technical:
                result["technical_analysis"] = await self._analyze_technical_structured(page, soup, response)
            
            # Links Analysis (Structured)
            result["links_analysis"] = await self._analyze_links_structured(soup, page.url)
            
            # Structured Data Analysis
            if options.include_structured_data:
                result["structured_data"] = await self._analyze_structured_data(soup)
            
            # Mobile Analysis
            if options.include_mobile:
                result["mobile_analysis"] = await self._analyze_mobile(soup)
            
            # Accessibility Analysis
            if options.include_accessibility:
                result["accessibility_analysis"] = await self._analyze_accessibility(soup)
            
            # External Resources
            if options.include_external_resources:
                result["external_resources"] = await self._analyze_external_resources(page.url)
            
            # ==========================================
            # PART 2: DETAILED ANALYSIS (Ultimate API)
            # ==========================================
            
            # Page Info (Detailed)
            result["page_info"] = self._collect_page_info(page.url, soup)
            
            # Business Information
            if options.include_business_info:
                result["business_info"] = await self._collect_business_info(soup, page.url)
            
            # Contact Information  
            result["contact_info"] = await self._collect_contact_info(soup)
            
            # Enhanced Content Analysis
            if options.include_content:
                result["content"] = await self._collect_enhanced_content(soup)
            
            # Enhanced Links Analysis
            result["links"] = await self._collect_enhanced_links(soup, page.url)
            
            # Enhanced Images Analysis
            if options.include_images:
                result["images"] = await self._collect_enhanced_images(soup, page.url)
            
            # Comprehensive Meta Data
            result["meta_data"] = await self._collect_meta_data(soup)
            
            # Page Structure
            result["page_structure"] = await self._collect_page_structure(soup)
            
            # SEO (Detailed)
            result["seo"] = await self._collect_enhanced_seo(soup, page.url)
            
            # Social Media
            result["social_media"] = await self._collect_social_media(soup)
            
            # Technical (Detailed)
            result["technical"] = await self._collect_technical_detailed(page, soup, response)
            
            # Robots.txt
            result["robots_txt"] = await self._collect_robots_txt(page.url)
            
            # Sitemap
            result["sitemap"] = await self._collect_sitemap_analysis(page.url)
            
            # ==========================================
            # SUMMARY & SCORES (Combined)
            # ==========================================
            
            # Calculate overall processing time
            processing_time = round(time.time() - start_time, 2)
            result["processing_time"] = processing_time
            
            # Generate combined summary
            result["summary"] = self._generate_combined_summary(result)
            
            logger.info(f"Combined analysis completed in {processing_time}s")
            return result
            
        except Exception as e:
            logger.error(f"Combined analysis error for {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'processing_time': time.time() - start_time
            }
        finally:
            if page and not page.is_closed():
                await page.close()
            if context:
                await browser_manager.close_context(context)
    
    async def _navigate_with_retry(self, page, url):
        """Navigate with retry logic"""
        try:
            return await page.goto(url, wait_until='networkidle', timeout=45000)
        except:
            try:
                return await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            except:
                return await page.goto(url, timeout=20000)
    
    # ==========================================
    # STRUCTURED ANALYSIS METHODS (Part 1)
    # ==========================================
    
    async def _analyze_performance(self, page, soup, response) -> Dict[str, Any]:
        """Performance analysis with Core Web Vitals"""
        
        # Get performance timing from browser
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
        
        # Simulate Core Web Vitals
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
        
        # CDN Detection
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
        """Estimate Largest Contentful Paint"""
        large_elements = soup.find_all(['img', 'video', 'h1', 'p'], limit=10)
        base_lcp = 1500
        if len(large_elements) > 5:
            base_lcp += 500
        return base_lcp
    
    async def _estimate_tti(self, soup, ttfb: float) -> float:
        """Estimate Time to Interactive"""
        script_tags = soup.find_all('script')
        js_complexity = len(script_tags) * 100
        return ttfb + js_complexity + 500
    
    async def _estimate_speed_index(self, soup) -> float:
        """Estimate Speed Index"""
        elements = len(soup.find_all())
        return 1000 + (elements * 2)
    
    def _detect_cdn(self, headers: Dict[str, str]) -> Optional[str]:
        """Detect CDN provider from headers"""
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
        """Analyze page resources"""
        css_links = soup.find_all('link', {'rel': 'stylesheet'})
        js_scripts = soup.find_all('script', {'src': True})
        images = soup.find_all('img')
        
        return {
            "css_resources": [{'url': link.get('href'), 'is_external': link.get('href', '').startswith('http'), 'media': link.get('media', 'all')} for link in css_links],
            "js_resources": [{'url': script.get('src'), 'is_external': script.get('src', '').startswith('http'), 'async': script.has_attr('async'), 'defer': script.has_attr('defer')} for script in js_scripts],
            "image_resources": [{'url': img.get('src'), 'alt': img.get('alt', ''), 'loading': img.get('loading', '')} for img in images],
            "external_css_count": len([link for link in css_links if link.get('href', '').startswith('http')]),
            "external_js_count": len([script for script in js_scripts if script.get('src', '').startswith('http')]),
            "total_resources": len(css_links) + len(js_scripts) + len(images)
        }
    
    def _calculate_performance_score(self, vitals: Dict, resources: Dict) -> int:
        """Calculate performance score"""
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
        """SEO analysis (structured format)"""
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
        
        headings = {
            'h1': soup.find_all('h1'),
            'h2': soup.find_all('h2'),
            'h3': soup.find_all('h3'),
            'h4': soup.find_all('h4'),
            'h5': soup.find_all('h5'),
            'h6': soup.find_all('h6')
        }
        
        heading_structure = {
            'h1_count': len(headings['h1']),
            'h2_count': len(headings['h2']),
            'h3_count': len(headings['h3']),
            'h4_count': len(headings['h4']),
            'h5_count': len(headings['h5']),
            'h6_count': len(headings['h6']),
            'proper_h1_usage': len(headings['h1']) == 1,
            'headings_by_level': {}
        }
        
        for level, tags in headings.items():
            heading_structure['headings_by_level'][level] = [
                {'text': tag.get_text().strip(), 'length': len(tag.get_text().strip())}
                for tag in tags
            ]
        
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
        """Calculate SEO score"""
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
        """Security analysis"""
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
        """Content analysis (structured format)"""
        text_content = soup.get_text()
        words = text_content.split()
        word_count = len(words)
        character_count = len(text_content)
        reading_time = max(1, word_count // 200)
        
        paragraphs = soup.find_all('p')
        paragraph_data = []
        for p in paragraphs[:20]:
            p_text = p.get_text().strip()
            if p_text:
                paragraph_data.append({
                    'text': p_text[:100] + '...' if len(p_text) > 100 else p_text,
                    'word_count': len(p_text.split()),
                    'has_links': bool(p.find_all('a'))
                })
        
        lists = soup.find_all(['ul', 'ol'])
        list_data = []
        for lst in lists:
            items = lst.find_all('li')
            list_data.append({
                'type': lst.name,
                'item_count': len(items),
                'items': [item.get_text().strip() for item in items[:5]]
            })
        
        tables = soup.find_all('table')
        table_data = []
        for table in tables:
            rows = table.find_all('tr')
            headers = table.find_all('th')
            table_data.append({
                'row_count': len(rows),
                'has_headers': len(headers) > 0,
                'header_count': len(headers)
            })
        
        return {
            "text_content": text_content[:500] + '...' if len(text_content) > 500 else text_content,
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
            "tables": {
                "total_tables": len(tables),
                "tables": table_data,
                "tables_with_headers": len([t for t in table_data if t['has_headers']])
            },
            "content_density": round(word_count / len(text_content) if text_content else 0, 3)
        }
    
    async def _analyze_images_structured(self, soup, base_url) -> Dict[str, Any]:
        """Image analysis (structured format)"""
        images = soup.find_all('img')
        
        image_data = []
        images_with_alt = 0
        lazy_loaded = 0
        responsive_images = 0
        format_distribution = {}
        
        for img in images:
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            if alt:
                images_with_alt += 1
            
            loading = img.get('loading', '')
            if loading == 'lazy':
                lazy_loaded += 1
            
            if img.get('srcset'):
                responsive_images += 1
            
            if src:
                ext = src.split('.')[-1].lower().split('?')[0]
                format_distribution[ext] = format_distribution.get(ext, 0) + 1
            
            image_data.append({
                'src': src,
                'alt': alt,
                'alt_length': len(alt),
                'has_alt': bool(alt),
                'title': img.get('title', ''),
                'width': img.get('width'),
                'height': img.get('height'),
                'loading': loading,
                'format': ext if src else 'unknown',
                'is_lazy_loaded': loading == 'lazy',
                'has_srcset': bool(img.get('srcset'))
            })
        
        return {
            "total_images": len(images),
            "images": image_data,
            "images_with_alt": images_with_alt,
            "images_without_alt": len(images) - images_with_alt,
            "lazy_loaded_images": lazy_loaded,
            "responsive_images": responsive_images,
            "format_distribution": format_distribution,
            "alt_text_quality": {
                "descriptive_alt": len([img for img in image_data if len(img['alt']) > 10]),
                "empty_alt": len([img for img in image_data if img['alt'] == '']),
                "missing_alt": len([img for img in image_data if not img['has_alt']]),
                "optimal_length_alt": len([img for img in image_data if 4 <= len(img['alt']) <= 125])
            }
        }
    
    async def _analyze_technical_structured(self, page, soup, response) -> Dict[str, Any]:
        """Technical analysis (structured format)"""
        headers = dict(response.headers) if response else {}
        
        html_validation = {
            "doctype": "html5" if soup.find('!DOCTYPE html') else "unknown",
            "lang_attribute": bool(soup.find('html', {'lang': True})),
            "charset_declared": bool(soup.find('meta', {'charset': True}))
        }
        
        external_stylesheets = len(soup.find_all('link', {'rel': 'stylesheet', 'href': lambda x: x and x.startswith('http')}))
        external_scripts = len(soup.find_all('script', {'src': lambda x: x and x.startswith('http')}))
        inline_styles = len(soup.find_all('style'))
        inline_scripts = len(soup.find_all('script', {'src': False}))
        
        html_content = str(soup)
        content_length = len(html_content.encode('utf-8'))
        
        return {
            "html_size": {
                "bytes": content_length,
                "kb": round(content_length / 1024, 2),
                "mb": round(content_length / 1024 / 1024, 2)
            },
            "response_headers": headers,
            "html_validation": html_validation,
            "resource_analysis": {
                "external_stylesheets": external_stylesheets,
                "external_scripts": external_scripts,
                "inline_styles": inline_styles,
                "inline_scripts": inline_scripts
            },
            "encoding": headers.get('content-encoding')
        }
    
    async def _analyze_links_structured(self, soup, base_url) -> Dict[str, Any]:
        """Links analysis (structured format)"""
        links = soup.find_all('a', href=True)
        
        internal_links = []
        external_links = []
        email_links = []
        phone_links = []
        nofollow_count = 0
        
        parsed_base = urlparse(base_url)
        
        for link in links:
            href = link.get('href', '')
            text = link.get_text().strip()
            title = link.get('title', '')
            rel = link.get('rel', [])
            
            if 'nofollow' in rel:
                nofollow_count += 1
            
            if href.startswith('mailto:'):
                email_links.append({'url': href, 'text': text, 'title': title})
            elif href.startswith('tel:'):
                phone_links.append({'url': href, 'text': text, 'title': title})
            elif href.startswith('http'):
                parsed_href = urlparse(href)
                if parsed_href.netloc == parsed_base.netloc:
                    internal_links.append({'url': href, 'text': text, 'title': title})
                else:
                    external_links.append({'url': href, 'text': text, 'title': title})
            else:
                internal_links.append({'url': urljoin(base_url, href), 'text': text, 'title': title})
        
        return {
            "total_links": len(links),
            "internal_links": {"count": len(internal_links), "links": internal_links[:20]},
            "external_links": {"count": len(external_links), "links": external_links[:20]},
            "email_links": {"count": len(email_links), "links": email_links},
            "phone_links": {"count": len(phone_links), "links": phone_links},
            "broken_link_indicators": 0,
            "nofollow_links": nofollow_count
        }
    
    async def _analyze_structured_data(self, soup) -> Dict[str, Any]:
        """Structured data analysis"""
        json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        json_ld_data = []
        schema_types = set()
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                json_ld_data.append(data)
                if isinstance(data, dict) and '@type' in data:
                    schema_types.add(data['@type'])
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and '@type' in item:
                            schema_types.add(item['@type'])
            except:
                continue
        
        microdata_elements = soup.find_all(attrs={'itemscope': True})
        
        og_tags = {}
        for meta in soup.find_all('meta'):
            property_attr = meta.get('property', '')
            if property_attr.startswith('og:'):
                og_tags[property_attr] = meta.get('content', '')
        
        twitter_tags = {}
        for meta in soup.find_all('meta'):
            name_attr = meta.get('name', '')
            if name_attr.startswith('twitter:'):
                twitter_tags[name_attr] = meta.get('content', '')
        
        return {
            "json_ld": json_ld_data,
            "microdata": microdata_elements,
            "opengraph": og_tags,
            "twitter_cards": twitter_tags,
            "schema_types": list(schema_types),
            "summary": {
                "has_structured_data": bool(json_ld_data or microdata_elements),
                "total_json_ld": len(json_ld_data),
                "total_microdata": len(microdata_elements),
                "total_schema_types": len(schema_types),
                "has_social_meta": bool(og_tags or twitter_tags)
            }
        }
    
    async def _analyze_mobile(self, soup) -> Dict[str, Any]:
        """Mobile analysis"""
        viewport_meta = soup.find('meta', {'name': 'viewport'})
        viewport_content = viewport_meta.get('content', '') if viewport_meta else ''
        
        media_queries_count = len(soup.find_all('style', string=lambda text: text and '@media' in text if text else False))
        apple_touch_icon = soup.find('link', {'rel': lambda x: x and 'apple-touch-icon' in x})
        
        return {
            "viewport_meta": {
                "exists": bool(viewport_meta),
                "content": viewport_content,
                "is_responsive": "width=device-width" in viewport_content
            },
            "mobile_specific_elements": {
                "apple_touch_icon": bool(apple_touch_icon),
                "mobile_meta_tags": 1 if apple_touch_icon else 0
            },
            "responsive_design_indicators": {
                "media_queries_in_css": media_queries_count,
                "has_viewport_meta": bool(viewport_meta)
            }
        }
    
    async def _analyze_accessibility(self, soup) -> Dict[str, Any]:
        """Accessibility analysis"""
        images = soup.find_all('img')
        images_with_alt = len([img for img in images if img.get('alt')])
        
        links = soup.find_all('a')
        links_with_text = len([link for link in links if link.get_text().strip()])
        
        h1_tags = soup.find_all('h1')
        forms = soup.find_all('form')
        labels = soup.find_all('label')
        
        aria_elements = soup.find_all(attrs={'aria-label': True})
        role_elements = soup.find_all(attrs={'role': True})
        
        html_tag = soup.find('html')
        has_lang = bool(html_tag and html_tag.get('lang'))
        
        return {
            "images": {
                "total_images": len(images),
                "images_with_alt": images_with_alt,
                "images_without_alt": len(images) - images_with_alt
            },
            "links": {
                "total_links": len(links),
                "links_with_text": links_with_text,
                "links_without_text": len(links) - links_with_text
            },
            "headings": {
                "h1_count": len(h1_tags),
                "proper_h1_usage": len(h1_tags) == 1
            },
            "forms": {
                "total_forms": len(forms),
                "total_labels": len(labels)
            },
            "aria_attributes": {
                "elements_with_aria_label": len(aria_elements),
                "elements_with_role": len(role_elements)
            },
            "language_declaration": {
                "html_has_lang": has_lang
            }
        }
    
    async def _analyze_external_resources(self, url: str) -> Dict[str, Any]:
        """External resources analysis"""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        robots_url = f"{base_url}/robots.txt"
        robots_data = await self._fetch_resource(robots_url)
        
        sitemap_url = f"{base_url}/sitemap.xml"
        sitemap_data = await self._fetch_resource(sitemap_url)
        
        return {
            "robots_txt": {
                "exists": robots_data['exists'],
                "content": robots_data['content'][:500] if robots_data['content'] else None,
                "size": len(robots_data['content']) if robots_data['content'] else 0
            },
            "sitemap": {
                "exists": sitemap_data['exists'],
                "size": len(sitemap_data['content']) if sitemap_data['content'] else 0,
                "content_type": sitemap_data.get('content_type', 'unknown')
            }
        }
    
    async def _fetch_resource(self, url: str) -> Dict[str, Any]:
        """Fetch external resource"""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            'exists': True,
                            'content': content,
                            'content_type': response.headers.get('content-type', '')
                        }
                    else:
                        return {'exists': False, 'content': None}
        except:
            return {'exists': False, 'content': None}
    
    # ==========================================
    # DETAILED ANALYSIS METHODS (Part 2)  
    # ==========================================
    
    def _collect_page_info(self, url, soup) -> Dict[str, Any]:
        """Detailed page information"""
        parsed_url = urlparse(url)
        title = soup.find('title')
        
        return {
            'url': url,
            'protocol': parsed_url.scheme,
            'domain': parsed_url.netloc.replace('www.', ''),
            'subdomain': 'www' if parsed_url.netloc.startswith('www.') else None,
            'path': parsed_url.path,
            'url_length': len(url),
            'is_ssl': parsed_url.scheme == 'https',
            'title': title.text.strip() if title else '',
            'title_length': len(title.text) if title else 0,
            'language': soup.find('html', lang=True).get('lang', 'unknown') if soup.find('html', lang=True) else 'unknown',
            'charset': self._extract_charset(soup)
        }
    
    def _extract_charset(self, soup):
        """Extract character encoding"""
        charset_meta = soup.find('meta', charset=True)
        if charset_meta:
            return charset_meta.get('charset', 'UTF-8')
        
        content_type_meta = soup.find('meta', {'http-equiv': 'content-type'})
        if content_type_meta:
            content = content_type_meta.get('content', '')
            if 'charset=' in content:
                return content.split('charset=')[1].split(';')[0].strip()
        
        return 'UTF-8'
    
    async def _collect_business_info(self, soup, url) -> Dict[str, Any]:
        """Business information extraction"""
        business_info = {
            'company_name': '',
            'addresses': [],
            'email_addresses': [],
            'phone_numbers': [],
            'social_profiles': []
        }
        
        title = soup.find('title')
        if title:
            business_info['company_name'] = title.text.strip()
        
        text_content = soup.get_text()
        
        # Dutch address patterns
        address_patterns = [
            r'\d{4}\s*[A-Z]{2}\s+[A-Za-z\s]+',
            r'[A-Za-z\s]+\s+\d+[A-Za-z]?\s*,\s*\d{4}\s*[A-Z]{2}',
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, text_content)
            business_info['addresses'].extend(matches[:5])
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text_content)
        business_info['email_addresses'] = list(set(emails))[:10]
        
        # Extract Dutch phone numbers
        phone_patterns = [
            r'\+31[\s\-]?\d{1,3}[\s\-]?\d{3}[\s\-]?\d{4}',
            r'0\d{1,3}[\s\-]?\d{3}[\s\-]?\d{4}',
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text_content)
            business_info['phone_numbers'].extend(phones)
        
        business_info['phone_numbers'] = list(set(business_info['phone_numbers']))[:5]
        
        return business_info
    
    async def _collect_contact_info(self, soup) -> Dict[str, Any]:
        """Contact information collection"""
        contact_pages = []
        contact_keywords = ['contact', 'over', 'about', 'info', 'support', 'help']
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').lower()
            text = link.get_text(strip=True).lower()
            
            if any(keyword in href or keyword in text for keyword in contact_keywords):
                contact_pages.append({
                    'url': link.get('href'),
                    'text': link.get_text(strip=True)
                })
        
        return {
            'contact_pages': contact_pages[:10]
        }
    
    async def _collect_enhanced_content(self, soup) -> Dict[str, Any]:
        """Enhanced content analysis"""
        text_content = soup.get_text()
        clean_text = re.sub(r'\s+', ' ', text_content).strip()
        words = clean_text.split()
        
        # Headings analysis
        headings = {}
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            if h_tags:
                headings[f'h{i}'] = []
                for h in h_tags[:10]:
                    heading_info = {
                        'text': h.get_text(strip=True),
                        'level': i,
                        'class': h.get('class', [])
                    }
                    headings[f'h{i}'].append(heading_info)
        
        # Paragraphs analysis
        paragraphs = []
        for p in soup.find_all('p')[:50]:
            text = p.get_text(strip=True)
            if len(text) > 10:
                paragraphs.append({
                    'text': text,
                    'length': len(text),
                    'word_count': len(text.split()),
                    'parent_tag': p.parent.name if p.parent else None
                })
        
        # Lists analysis
        lists = []
        for ul in soup.find_all(['ul', 'ol'])[:20]:
            list_items = ul.find_all('li')
            list_info = {
                'type': ul.name,
                'total_items': len(list_items),
                'class': ul.get('class', []),
                'items': []
            }
            
            for li in list_items[:10]:
                list_info['items'].append({
                    'text': li.get_text(strip=True),
                    'has_links': bool(li.find('a'))
                })
            
            lists.append(list_info)
        
        # Text blocks analysis
        text_blocks = []
        for section in soup.find_all(['section', 'article', 'div', 'main'])[:20]:
            text = section.get_text(strip=True)
            if len(text) > 100:
                text_blocks.append({
                    'tag': section.name,
                    'class': section.get('class', []),
                    'text': text[:500] + '...' if len(text) > 500 else text,
                    'word_count': len(text.split())
                })
        
        nav_content = ""
        nav = soup.find('nav')
        if nav:
            nav_content = nav.get_text(strip=True)
        
        footer_content = ""
        footer = soup.find('footer')
        if footer:
            footer_content = footer.get_text(strip=True)
        
        return {
            'word_count': len(words),
            'reading_time': max(1, len(words) // 200),
            'text_content': clean_text,
            'text_content_truncated': len(clean_text) > 10000,
            'text_density': len(clean_text) / len(str(soup)) if len(str(soup)) > 0 else 0,
            'headings': headings,
            'paragraphs': paragraphs,
            'paragraphs_total': len(soup.find_all('p')),
            'lists': lists,
            'text_blocks': text_blocks[:10],
            'navigation_content': nav_content,
            'footer_content': footer_content,
            'language': 'nl-NL'
        }
    
    async def _collect_enhanced_links(self, soup, base_url) -> Dict[str, Any]:
        """Enhanced links analysis with categorization"""
        links = {
            'all': [],
            'internal': [],
            'external': [],
            'email': [],
            'navigation': [],
            'footer': []
        }
        
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            link_info = {
                'url': href,
                'text': text,
                'class': link.get('class', []),
                'parent_element': link.parent.name if link.parent else None
            }
            
            links['all'].append(link_info)
            
            if href.startswith('mailto:'):
                links['email'].append({
                    'email': href[7:],
                    'text': text
                })
            elif href.startswith('http'):
                link_domain = urlparse(href).netloc
                if link_domain == base_domain:
                    links['internal'].append(link_info)
                else:
                    links['external'].append(link_info)
            else:
                full_url = urljoin(base_url, href)
                link_info['url'] = full_url
                links['internal'].append(link_info)
            
            parent_nav = link.find_parent('nav')
            parent_footer = link.find_parent('footer')
            
            if parent_nav:
                links['navigation'].append(link_info)
            elif parent_footer:
                links['footer'].append(link_info)
        
        return {
            'all': links['all'][:50],
            'internal': links['internal'][:30],
            'external': links['external'][:20],
            'email': links['email'][:10],
            'navigation': links['navigation'][:15],
            'footer': links['footer'][:20],
            'internal_links_count': len(links['internal']),
            'external_links_count': len(links['external']),
            'email_count': len(links['email'])
        }
    
    async def _collect_enhanced_images(self, soup, base_url) -> List[Dict[str, Any]]:
        """Enhanced images analysis"""
        images = []
        
        for img in soup.find_all('img')[:30]:
            src = img.get('src', '')
            if src:
                absolute_url = urljoin(base_url, src) if not src.startswith('http') else src
                format_ext = src.split('.')[-1].lower() if '.' in src else 'unknown'
                
                alt_text = img.get('alt', '')
                alt_quality = 'good' if len(alt_text) > 10 else 'poor' if alt_text else 'missing'
                
                image_info = {
                    'src': absolute_url,
                    'alt': alt_text,
                    'alt_length': len(alt_text),
                    'alt_quality': alt_quality,
                    'width': img.get('width', ''),
                    'height': img.get('height', ''),
                    'loading': img.get('loading', ''),
                    'class': img.get('class', []),
                    'parent_element': img.parent.name if img.parent else None,
                    'format': format_ext,
                    'has_lazy_loading': img.get('loading') == 'lazy'
                }
                
                images.append(image_info)
        
        return images[:15]
    
    async def _collect_meta_data(self, soup) -> Dict[str, Any]:
        """Comprehensive meta data collection"""
        meta_data = {}
        
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            
            if name and content:
                meta_data[name.lower()] = content
        
        # Stylesheets
        stylesheets = []
        for link in soup.find_all('link', rel='stylesheet'):
            stylesheet_info = {
                'href': link.get('href'),
                'media': link.get('media', 'all'),
                'type': link.get('type', 'text/css'),
                'is_external': link.get('href', '').startswith('http')
            }
            stylesheets.append(stylesheet_info)
        
        meta_data['stylesheets'] = stylesheets
        
        # External scripts
        external_scripts = []
        for script in soup.find_all('script', src=True):
            script_info = {
                'src': script.get('src'),
                'type': script.get('type', 'text/javascript'),
                'defer': script.has_attr('defer'),
                'async': script.has_attr('async'),
                'is_external': script.get('src', '').startswith('http')
            }
            external_scripts.append(script_info)
        
        meta_data['external_scripts'] = external_scripts
        
        # Favicons
        favicons = []
        for link in soup.find_all('link', rel=lambda x: x and 'icon' in x.lower()):
            favicon_info = {
                'href': link.get('href'),
                'rel': link.get('rel'),
                'sizes': link.get('sizes'),
                'type': link.get('type')
            }
            favicons.append(favicon_info)
        
        meta_data['favicons'] = favicons
        
        # Alternate languages
        alternate_languages = []
        for link in soup.find_all('link', rel='alternate', hreflang=True):
            alternate_languages.append({
                'hreflang': link.get('hreflang'),
                'href': link.get('href')
            })
        
        meta_data['alternate_languages'] = alternate_languages
        
        return meta_data
    
    async def _collect_page_structure(self, soup) -> Dict[str, Any]:
        """Page structure analysis"""
        semantic_elements = []
        semantic_tags = ['nav', 'header', 'main', 'section', 'article', 'aside', 'footer']
        
        for tag in semantic_tags:
            count = len(soup.find_all(tag))
            if count > 0:
                semantic_elements.append({
                    'tag': tag,
                    'count': count
                })
        
        total_elements = len(soup.find_all())
        has_nav = bool(soup.find('nav'))
        has_footer = bool(soup.find('footer'))
        
        navigation_items = 0
        nav = soup.find('nav')
        if nav:
            navigation_items = len(nav.find_all('a'))
        
        content_sections = len(soup.find_all(['section', 'article']))
        
        return {
            'total_elements': total_elements,
            'semantic_elements': semantic_elements,
            'has_nav': has_nav,
            'has_footer': has_footer,
            'navigation_items': navigation_items,
            'content_sections': content_sections
        }
    
    async def _collect_enhanced_seo(self, soup, url) -> Dict[str, Any]:
        """Enhanced SEO analysis"""
        title = soup.find('title')
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        canonical = soup.find('link', rel='canonical')
        robots_meta = soup.find('meta', attrs={'name': 'robots'})
        
        h1_count = len(soup.find_all('h1'))
        h2_count = len(soup.find_all('h2'))
        h3_count = len(soup.find_all('h3'))
        total_headings = sum(len(soup.find_all(f'h{i}')) for i in range(1, 7))
        
        h1_text = [h1.get_text(strip=True) for h1 in soup.find_all('h1')]
        
        images = soup.find_all('img')
        images_total = len(images)
        
        base_domain = urlparse(url).netloc
        internal_links = 0
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href.startswith('/') or base_domain in href:
                internal_links += 1
        
        text_content = soup.get_text()
        words = text_content.split()
        word_count_estimate = len(words)
        
        html_size = len(str(soup))
        text_size = len(text_content)
        text_to_html_ratio = text_size / html_size if html_size > 0 else 0
        
        structured_data_present = bool(soup.find('script', type='application/ld+json'))
        opengraph_present = bool(soup.find('meta', property=lambda x: x and x.startswith('og:')))
        twitter_cards_present = bool(soup.find('meta', attrs={'name': lambda x: x and x.startswith('twitter:')}))
        
        return {
            'title_text': title.text.strip() if title else '',
            'title_length': len(title.text) if title else 0,
            'title_words': len(title.text.split()) if title else 0,
            'meta_description_text': meta_desc.get('content', '') if meta_desc else '',
            'meta_description_length': len(meta_desc.get('content', '')) if meta_desc else 0,
            'meta_description_words': len(meta_desc.get('content', '').split()) if meta_desc else 0,
            'meta_keywords': meta_keywords.get('content', '') if meta_keywords else '',
            'canonical_url': canonical.get('href', '') if canonical else '',
            'robots_meta': robots_meta.get('content', '') if robots_meta else '',
            'h1_count': h1_count,
            'h1_text': h1_text,
            'h2_count': h2_count,
            'h3_count': h3_count,
            'total_headings': total_headings,
            'images_total': images_total,
            'internal_links_count': internal_links,
            'word_count_estimate': word_count_estimate,
            'text_to_html_ratio': round(text_to_html_ratio, 3),
            'structured_data_present': structured_data_present,
            'opengraph_present': opengraph_present,
            'twitter_cards_present': twitter_cards_present
        }
    
    async def _collect_social_media(self, soup) -> Dict[str, Any]:
        """Social media information collection"""
        open_graph = {}
        for meta in soup.find_all('meta'):
            property_attr = meta.get('property', '')
            if property_attr.startswith('og:'):
                open_graph[property_attr.replace('og:', '')] = meta.get('content', '')
        
        twitter_cards = {}
        for meta in soup.find_all('meta'):
            name_attr = meta.get('name', '')
            if name_attr.startswith('twitter:'):
                twitter_cards[name_attr.replace('twitter:', '')] = meta.get('content', '')
        
        total_social_tags = len(open_graph) + len(twitter_cards)
        
        return {
            'open_graph': open_graph,
            'twitter_cards': twitter_cards,
            'summary': {
                'has_open_graph': bool(open_graph),
                'has_twitter_cards': bool(twitter_cards),
                'total_social_tags': total_social_tags
            }
        }
    
    async def _collect_technical_detailed(self, page, soup, response) -> Dict[str, Any]:
        """Detailed technical analysis"""
        try:
            performance_metrics = await page.evaluate("""
                () => {
                    const timing = performance.timing;
                    return {
                        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        loadTime: timing.loadEventEnd - timing.navigationStart,
                        firstPaint: performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || null,
                        firstContentfulPaint: performance.getEntriesByType('paint').find(p => p.name === 'first-contentful-paint')?.startTime || null
                    };
                }
            """)
        except:
            performance_metrics = {}
        
        response_headers = {}
        if response:
            response_headers = dict(response.headers)
        
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        mobile_friendly = {
            'has_viewport': bool(viewport_meta),
            'viewport_content': viewport_meta.get('content', '') if viewport_meta else '',
            'mobile_specific_meta': bool(soup.find('meta', attrs={'name': 'mobile-web-app-capable'})),
            'responsive_meta_tags': len(soup.find_all('meta', attrs={'name': lambda x: x and 'mobile' in x.lower()})),
            'touch_icons': len(soup.find_all('link', rel=lambda x: x and 'apple-touch-icon' in x)),
            'media_queries_in_html': len(re.findall(r'@media', str(soup)))
        }
        
        page_speed_insights = {
            'images_total': len(soup.find_all('img')),
            'images_with_alt': len([img for img in soup.find_all('img') if img.get('alt')]),
            'images_lazy_loading': len([img for img in soup.find_all('img') if img.get('loading') == 'lazy']),
            'external_scripts': len(soup.find_all('script', src=lambda x: x and x.startswith('http'))),
            'inline_scripts': len(soup.find_all('script', src=False)),
            'external_stylesheets': len(soup.find_all('link', rel='stylesheet', href=lambda x: x and x.startswith('http'))),
            'inline_styles': len(soup.find_all('style')),
            'total_links': len(soup.find_all('a', href=True)),
            'svg_elements': len(soup.find_all('svg'))
        }
        
        accessibility = {
            'images_with_alt': len([img for img in soup.find_all('img') if img.get('alt')]),
            'form_inputs': len(soup.find_all(['input', 'textarea', 'select'])),
            'aria_labels': len(soup.find_all(attrs={'aria-label': True})),
            'role_attributes': len(soup.find_all(attrs={'role': True})),
            'h1_count': len(soup.find_all('h1')),
            'headings_structure': sum(len(soup.find_all(f'h{i}')) for i in range(1, 7)),
            'lang_attribute': bool(soup.find('html', lang=True))
        }
        
        html_content = str(soup)
        html_size = len(html_content.encode('utf-8'))
        
        return {
            'performance': performance_metrics,
            'response_headers': response_headers,
            'mobile_friendly': mobile_friendly,
            'page_speed_insights': page_speed_insights,
            'accessibility': accessibility,
            'html_size': html_size,
            'html_size_kb': round(html_size / 1024, 2),
            'doctype': 'html5' if '<!DOCTYPE html>' in html_content else 'other',
            'security': {
                'https': page.url.startswith('https://')
            }
        }
    
    async def _collect_robots_txt(self, url) -> Dict[str, Any]:
        """Robots.txt analysis"""
        try:
            robots_url = urljoin(url, '/robots.txt')
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            'url': robots_url,
                            'status': response.status,
                            'content': content,
                            'content_type': response.headers.get('content-type', ''),
                            'size': len(content)
                        }
                    else:
                        return {
                            'url': robots_url,
                            'status': response.status,
                            'content': None
                        }
        except:
            return {
                'url': urljoin(url, '/robots.txt'),
                'status': 0,
                'content': None
            }
    
    async def _collect_sitemap_analysis(self, url) -> List[Dict[str, Any]]:
        """Sitemap analysis"""
        sitemaps = []
        sitemap_urls = [
            urljoin(url, '/sitemap.xml'),
            urljoin(url, '/sitemap_index.xml'),
            urljoin(url, '/sitemap-index.xml')
        ]
        
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for sitemap_url in sitemap_urls:
                try:
                    async with session.get(sitemap_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            sitemap_info = {
                                'url': sitemap_url,
                                'status': response.status,
                                'content_type': response.headers.get('content-type', ''),
                                'size': len(content),
                                'is_compressed': 'gzip' in response.headers.get('content-encoding', ''),
                                'has_images': '<image:' in content,
                                'url_count': content.count('<url>'),
                                'urls': []
                            }
                            
                            # Parse XML to extract URLs
                            try:
                                import xml.etree.ElementTree as ET
                                root = ET.fromstring(content.encode())
                                
                                ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                                
                                for url_elem in root.findall('.//sitemap:url', ns)[:20]:
                                    loc = url_elem.find('sitemap:loc', ns)
                                    lastmod = url_elem.find('sitemap:lastmod', ns)
                                    changefreq = url_elem.find('sitemap:changefreq', ns)
                                    priority = url_elem.find('sitemap:priority', ns)
                                    
                                    url_info = {
                                        'type': 'url',
                                        'url': loc.text if loc is not None else '',
                                        'lastmod': lastmod.text if lastmod is not None else '',
                                        'changefreq': changefreq.text if changefreq is not None else '',
                                        'priority': priority.text if priority is not None else ''
                                    }
                                    sitemap_info['urls'].append(url_info)
                            except:
                                pass
                            
                            sitemaps.append(sitemap_info)
                            break
                            
                except:
                    continue
        
        return sitemaps
    
    def _generate_combined_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate combined summary from both analysis types"""
        
        scores = {}
        
        # Extract scores from structured analysis
        if 'performance' in analysis_results:
            scores['performance_score'] = analysis_results['performance'].get('performance_score', 0)
        
        if 'seo_analysis' in analysis_results:
            scores['seo_score'] = analysis_results['seo_analysis'].get('seo_score', 0)
        
        if 'security_analysis' in analysis_results:
            scores['security_score'] = analysis_results['security_analysis'].get('security_score', 0)
        
        # Calculate additional scores from detailed analysis
        if 'accessibility_analysis' in analysis_results:
            acc_data = analysis_results['accessibility_analysis']
            acc_score = 100
            if acc_data['images']['images_without_alt'] > 0:
                acc_score -= 20
            if not acc_data['language_declaration']['html_has_lang']:
                acc_score -= 10
            if acc_data['headings']['h1_count'] != 1:
                acc_score -= 15
            scores['accessibility_score'] = max(acc_score, 0)
        
        if 'mobile_analysis' in analysis_results:
            mobile_data = analysis_results['mobile_analysis']
            mobile_score = 100
            if not mobile_data['viewport_meta']['exists']:
                mobile_score -= 30
            if not mobile_data['viewport_meta']['is_responsive']:
                mobile_score -= 20
            scores['mobile_score'] = max(mobile_score, 0)
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0
        
        # Generate recommendations
        recommendations = []
        
        if scores.get('performance_score', 0) < 80:
            recommendations.append("Improve page loading speed and Core Web Vitals")
        if scores.get('seo_score', 0) < 80:
            recommendations.append("Optimize meta tags and heading structure")
        if scores.get('security_score', 0) < 80:
            recommendations.append("Add missing security headers")
        if scores.get('accessibility_score', 0) < 80:
            recommendations.append("Improve accessibility with alt texts and proper markup")
        if scores.get('mobile_score', 0) < 80:
            recommendations.append("Ensure responsive design and mobile optimization")
        
        return {
            "overall_scores": {
                **scores,
                "overall_score": round(overall_score, 1)
            },
            "recommendations": recommendations,
            "analysis_summary": {
                "pages_analyzed": 1,
                "total_issues": len(recommendations),
                "critical_issues": len([r for r in recommendations if "security" in r.lower()]),
                "performance_issues": len([r for r in recommendations if "speed" in r.lower() or "performance" in r.lower()])
            },
            "data_coverage": {
                "has_structured_analysis": 'seo_analysis' in analysis_results,
                "has_detailed_analysis": 'content' in analysis_results,
                "has_business_info": 'business_info' in analysis_results,
                "has_technical_details": 'technical' in analysis_results,
                "total_data_points": len(analysis_results)
            }
        }

# =====================
# API ENDPOINTS
# =====================

combined_analyzer = CombinedWebsiteAnalyzer()

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "features": "Combined structured + detailed analysis with fixed browser management"
    }

@app.post("/analyze/combined")
async def analyze_website_combined(request: CombinedAnalyzeRequest):
    """
    Combined website analysis - provides BOTH structured metrics AND detailed content analysis
    
    This endpoint combines:
    - Structured analysis: Performance metrics, SEO scores, security analysis 
    - Detailed analysis: Business info, enhanced content structure, comprehensive meta data
    
    Returns the most complete website analysis available.
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
    Quick combined analysis via GET request
    """
    # Ensure URL has protocol
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Validate URL format
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

@app.get("/")
async def root():
    return {
        "message": "Combined Ultimate Website Analyzer API with Fixed Browser Management",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "status": "operational",
        "description": "Complete website analysis combining both structured metrics AND detailed content analysis with optimized browser management for sequential scans",
        "enhancements": [
            "🔧 Fixed Browser Management for efficient resource usage",
            "🧹 Automatic memory cleanup and browser recreation",
            "🔄 Fresh browser every 2 requests",
            "📊 Reduced memory footprint and improved stability",
            "⚡ Optimized for 5+ sequential scans"
        ],
        "features": [
            "🚀 Performance Analysis (Core Web Vitals, Resource Analysis)",
            "🔍 Structured SEO Analysis (Scores, Recommendations)",
            "🛡️ Security Analysis (Headers, SSL, Vulnerabilities)",
            "📊 Enhanced Content Analysis (Detailed Structure, Text Blocks)",
            "🏢 Business Information Extraction",
            "📞 Contact Information Detection", 
            "🔗 Enhanced Links Categorization (Navigation, Footer, etc.)",
            "🖼️ Comprehensive Image Analysis",
            "📱 Mobile/Responsive Analysis",
            "♿ Accessibility Analysis",
            "🏗️ Technical Analysis (Performance, HTML Validation)",
            "🗂️ Page Structure Analysis",
            "📋 Comprehensive Meta Data Collection",
            "🌐 Social Media Optimization Analysis",
            "🤖 Robots.txt & Sitemap Analysis",
            "📈 Combined Scoring & Recommendations"
        ],
        "main_endpoint": {
            "POST /analyze/combined": "Full combined analysis with all options",
            "GET /analyze/combined/{url}": "Quick combined analysis of URL"
        },
        "output_structure": {
            "structured_data": "Performance scores, SEO metrics, security analysis",
            "detailed_data": "Business info, enhanced content, comprehensive meta data",
            "combined_summary": "Overall scores and recommendations from both analyses"
        },
        "limits": {
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_workers": MAX_WORKERS,
            "browser_recreation_interval": "Every 2 requests",
            "memory_cleanup": "Every request"
        }
    }

# =====================
# STARTUP & CLEANUP
# =====================

@app.on_event("startup")
async def startup_event():
    """Initialize browser on startup"""
    logger.info("Combined Ultimate Website Analyzer API with Fixed Browser Management starting up...")
    try:
        # Pre-initialize browser for faster first request
        context = await browser_manager.get_context()
        await browser_manager.close_context(context)
        logger.info("Browser manager initialized successfully")
    except Exception as e:
        logger.error(f"Browser manager initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down Combined Ultimate Website Analyzer API...")
    await browser_manager.close()
    # Force final garbage collection
    gc.collect()
    logger.info("Cleanup completed")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render uses dynamic ports)
    port = int(os.getenv("PORT", 8000))
    
    # Production settings
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
        # Development settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="debug"
        )
