class CacheManager:
    _instance = None

    def __init__(self):
        self.cache = {}

    def get(self, name: str, func):
        if name not in self.cache:
            print("Creating cached object '%s'" % name)
            self.cache[name] = func()

        return self.cache[name]

    @staticmethod
    def instance() -> "CacheManager":
        if CacheManager._instance is None:
            CacheManager._instance = CacheManager()

        return CacheManager._instance
