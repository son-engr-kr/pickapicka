#!/bin/bash
# Pickapicka {{VERSION}} for macOS, installed from Terminal:
#
#   curl -fsSL {{SITE_URL}}install.sh | bash
#
# A .pkg downloaded in a browser carries macOS's quarantine mark, and since the
# package is not signed by Apple, macOS will not open it until it is allowed in
# System Settings -> Privacy & Security. curl adds no such mark, so the same
# package installs at once. It is checked against the SHA-256 that GitHub
# published for it, as the app's own updater does (updater.py), and installed
# with macOS's own installer, which asks for the Mac's password.
#
# site/build.py fills in the version, the link and the checksum of each
# release; this file in the repository is the template. Everything it runs
# comes with macOS.
set -euo pipefail

VERSION="{{VERSION}}"
PKG_URL="{{MAC_URL}}"
PKG_SHA256="{{MAC_SHA256}}"
APP="/Applications/Pickapicka.app"

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }
fail() { printf '\n\033[1;31m%s\033[0m\n\n' "$1" >&2; exit 1; }

# All of it in one function, called on the last line: piped into bash, a
# download cut short runs nothing rather than half of it.
main() {
  [ "$(uname -s)" = Darwin ] || fail "This installer is for macOS. Other systems: {{SITE_URL}}#download"
  # Apple Silicon, also from a Terminal running under Rosetta.
  [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ] \
    || fail "Pickapicka needs a Mac with Apple Silicon (M1 or later)."
  local major
  major=$(sw_vers -productVersion | cut -d. -f1)
  [ "$major" -ge 12 ] || fail "Pickapicka needs macOS 12 or later. This Mac has $(sw_vers -productVersion)."

  local pkg got
  # Global, as the trap runs after main has returned.
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  pkg="$tmp/Pickapicka-$VERSION.pkg"

  say "Downloading Pickapicka $VERSION ({{MAC_SIZE}})..."
  curl -fL --progress-bar -o "$pkg" "$PKG_URL" \
    || fail "The download failed. Check the internet connection and run the line again."

  got=$(shasum -a 256 "$pkg" | cut -d' ' -f1)
  [ "$got" = "$PKG_SHA256" ] \
    || fail "The download is not the file GitHub published (its checksum differs), so it was not installed. Run the line again."

  say "Installing. Type the password you log in to this Mac with, then press Return."
  printf '(Nothing appears while you type it. That is normal. The first time, macOS\n'
  printf 'prints a WARNING about sudo before it asks: its usual note, not a problem.)\n'
  sudo installer -pkg "$pkg" -target / >/dev/null \
    || fail "The install did not finish. Run the line again and check the password. It has to be an administrator account of this Mac."

  say "Pickapicka $VERSION is installed in Applications. Opening it..."
  open "$APP"
  printf 'Next time, open it from Launchpad or the Applications folder.\nThis Terminal window can be closed.\n\n'
}

main "$@"
