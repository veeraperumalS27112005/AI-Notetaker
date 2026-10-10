"""
One-time setup: downloads Argos Translate language packages.
Run this ONCE before starting the app. Internet needed for download only.
After this, translation works 100% offline forever.
"""
import argostranslate.package
import argostranslate.translate

# Languages the app supports (English -> each)
LANG_PAIRS = [
    ("en", "ta"),  # Tamil     ← most important
    ("en", "hi"),  # Hindi
    ("en", "es"),  # Spanish
    ("en", "fr"),  # French
    ("en", "de"),  # German
    ("en", "ja"),  # Japanese
    ("en", "ar"),  # Arabic
]

print("Fetching available packages from Argos Translate index...")
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()

# Get already installed pairs using the packages API (compatible with newer argostranslate)
installed_pkgs = argostranslate.package.get_installed_packages()
installed_codes = {(p.from_code, p.to_code) for p in installed_pkgs}

print(f"Found {len(available)} packages. Installing needed ones...\n")

for from_code, to_code in LANG_PAIRS:
    if (from_code, to_code) in installed_codes:
        print(f"  [OK] {from_code} -> {to_code} already installed, skipping.")
        continue

    pkg = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None
    )
    if pkg:
        print(f"  [DOWNLOADING] Installing {from_code} -> {to_code} ({pkg.package_version})...")
        argostranslate.package.install_from_path(pkg.download())
        print(f"  [OK] {from_code} -> {to_code} installed!")
    else:
        print(f"  [!] Package {from_code} -> {to_code} not found in index. Skipping.")

print("\n[OK] All done! Argos Translate is ready for offline use.")
print("   You can now run record_and_summarize.py — Tamil translation will work offline.")

# Quick test
print("\n[Test] Quick test (en -> ta):")
try:
    langs = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in langs if l.code == "en"), None)
    to_lang   = next((l for l in langs if l.code == "ta"), None)
    if from_lang and to_lang:
        t = from_lang.get_translation(to_lang)
        result = t.translate("Hello! This is a test of offline Tamil translation.")
        print(f"   Input : Hello! This is a test of offline Tamil translation.")
        print(f"   Output: {result}")
    else:
        print("   Tamil package not found — check install above.")
except Exception as e:
    print(f"   Test failed: {e}")
