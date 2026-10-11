import json
import os
from PyQt6.QtCore import QObject, pyqtSignal

class I18nManager(QObject):
    """Nyelvi szótárat kezelő osztály JSON alapon."""
    language_changed = pyqtSignal()

    def __init__(self, locales_dir="src/locales", default_lang="hu"):
        super().__init__()
        self.locales_dir = locales_dir
        self.current_lang = default_lang
        self.translations = {}
        self.load_language(default_lang)

    def load_language(self, lang_code: str) -> bool:
        """Betölti a megadott nyelvi JSON fájlt."""
        filepath = os.path.join(self.locales_dir, f"{lang_code}.json")
        if not os.path.exists(filepath):
            print(f"⚠️ Nyelvi fájl nem található: {filepath}")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.translations = json.load(f)
            self.current_lang = lang_code
            self.language_changed.emit()
            return True
        except Exception as e:
            print(f"❌ Hiba a nyelvi fájl betöltésekor [{filepath}]: {e}")
            return False

    def t(self, key: str, default: str = "") -> str:
        """Visszaadja a kulcshoz tartozó fordítást."""
        return self.translations.get(key, default if default else key)

# Globális singleton példány a könnyű eléréshez
i18n = I18nManager()