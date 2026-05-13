[app]

# (str) Title of your application
title = SSQ Predictor

# Release signing (JKS keystore generated during build)
p4a.release_keyalias = ssqkey
p4a.release_keystore = ssq-release.jks
p4a.release_keystore_pass = ssq2026predictor
p4a.release_keyalias_pass = ssq2026predictor

# (str) Package name
package.name = ssqpredictor

# (str) Package domain (needed for android/ios packaging)
package.domain = org.ssqpredictor

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all)
source.include_exts = py,png,jpg,kv,atlas,json

# (list) List of inclusions using pattern matching
source.include_patterns = *.py, *.json, *.kv

# (list) Source files to exclude
source.exclude_exts = spec

# (list) List of directories to exclude
source.exclude_dirs = __pycache__, data_backups, .claude

# (str) Application versioning
version = 1.0

# (str) Application orientation (landscape/portrait or sensor)
orientation = portrait

# (list) Permissions
android.permissions = INTERNET

# (int) Target Android API
android.api = 34

# (int) Minimum API
android.minapi = 21

# (int) Android SDK version
android.sdk = 34

# (str) Android NDK version
android.ndk = 25b

# (bool) Use AndroidX
android.use_androidx = True

# (str) Bootstrap for Android
p4a.bootstrap = sdl2

# (list) Pattern to whitelist for the whole project
p4a.whitelist_src =

# (list) Requirements
requirements = python3,kivy,requests

# (str) Supported Python versions
p4a.python_version = 3

# (str) Presplash of the application
presplash.filename = %(source.dir)s/presplash.png

# (str) Icon of the application
icon.filename = %(source.dir)s/icon.png

# (list) Platforms to build
p4a.commands = build

# (str) The Android arch to build for
android.arch = arm64-v8a

# (str) The log level
log_level = 2

# (int) Warn on permissions
warn_on_permissions = 1

# (str) Custom gradle dependencies
#android.gradle_dependencies =

# (list) Java classes to add as activities
#android.add_activities =

# (str) OUYA Console category
#ouya.category = NONE

# (str) Which.NET framework to target
#android.meta_data =

# (str) URL scheme
#android.url_scheme =

# (str) Custom AndroidManifest.xml
#android.manifest.custom =

# (str) Activity that acts as the application entry
android.entrypoint = org.kivy.android.PythonActivity

# (bool) Indicate if the application should be fullscreen
fullscreen = 1

# (str) Supported orientation
android.orientation = portrait

# (bool) Enable AndroidX
android.enable_androidx = True
