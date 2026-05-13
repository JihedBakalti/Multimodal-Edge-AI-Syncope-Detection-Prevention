"""
Cross-origin SPA (e.g. Vercel) calling this API on Render cannot send Django's CSRF
cookie/token on POST. Exempt /api/ from CSRF enforcement; rely on CORS + auth classes.
"""


class ExemptApiFromCsrfMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith("/api/"):
            request._dont_enforce_csrf_checks = True
        return self.get_response(request)
