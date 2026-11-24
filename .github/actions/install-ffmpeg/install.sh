#!/usr/bin/env bash
# Установка ffmpeg в CI: один путь для jobs checks и clean-install (#183).
#   install.sh key      — ключ кэша набора .deb в $GITHUB_OUTPUT.
#   install.sh install  — CACHE_HIT=true: набор из $FFMPEG_DEBS через dpkg -i, без apt-get и без
#                         зеркала apt. Иначе — скачивание с ограниченными повторами (#65, #70),
#                         установка и копия .deb установленных версий в $FFMPEG_DEBS для actions/cache/save.
set -euo pipefail

debs=${FFMPEG_DEBS:?FFMPEG_DEBS is not set}
archives=${APT_ARCHIVES:-/var/cache/apt/archives}

key() {
  local image level
  if [ -n "${ImageOS:-}" ] && [ -n "${ImageVersion:-}" ]; then
    image="$ImageOS-$ImageVersion"
  else
    echo "::warning::ImageOS or ImageVersion is not set: the ffmpeg package cache serves this run only"
    image="unknown-image-run-${GITHUB_RUN_ID:-0}"
  fi
  # apt берёт сборки под уровень CPU (amd64v3): набор с одного хоста не годится хосту ниже уровнем.
  level=$(/lib64/ld-linux-x86-64.so.2 --help 2>/dev/null |
    sed -n 's/^ *\(x86-64-v[0-9]\) (supported.*/\1/p' | sed -n 1p) || true
  local script_sha
  script_sha=$(sha256sum "${BASH_SOURCE[0]}" | cut -c1-16)
  echo "key=ffmpeg-debs-$image-$(dpkg --print-architecture)-${level:-cpu-unknown}-$script_sha" >>"$GITHUB_OUTPUT"
}

report() {
  local version
  version=$(ffmpeg -version)
  echo "${version%%$'\n'*}"
  echo "ffmpeg installed in ${SECONDS} s ($1)"
}

install_from_cache() {
  shopt -s nullglob
  local files=("$debs"/*.deb) packages=("$debs"/ffmpeg_*.deb)
  if [ "${#packages[@]}" -ne 1 ]; then
    echo "::error::the ffmpeg package cache has no single ffmpeg_*.deb (${#files[@]} files); delete the cache entry with gh cache delete and rerun"
    exit 1
  fi
  echo "ffmpeg package cache hit: dpkg -i of ${#files[@]} cached .deb files ($(basename "${packages[0]}")); no apt-get update, nothing downloaded from the apt mirror"
  if ! sudo dpkg -i "${files[@]}"; then
    echo "::error::the cached ffmpeg packages did not install with dpkg -i; delete the cache entry with gh cache delete and rerun"
    exit 1
  fi
  report "cache hit"
}

download_and_install() {
  echo "ffmpeg package cache miss: download from the apt mirror with bounded retry"
  apt_opts=(-q -o Acquire::Retries=3 -o Acquire::http::Timeout=30)
  attempts=3
  attempt_minutes=5
  # Набор для кэша — ровно то, что скачает эта установка. .deb, скачанные попыткой, которую оборвал
  # timeout, остаются в кэше apt, и следующая попытка продолжает с них.
  sudo apt-get "${apt_opts[@]}" clean
  for attempt in $(seq "$attempts"); do
    if sudo timeout --kill-after=30s "${attempt_minutes}m" bash -c \
      'apt-get "$@" update && apt-get "$@" install -y --download-only ffmpeg' apt "${apt_opts[@]}"; then
      break
    fi
    echo "::warning::ffmpeg download attempt $attempt of $attempts failed or exceeded ${attempt_minutes} min"
    if [ "$attempt" -eq "$attempts" ]; then
      echo "::error::ffmpeg packages were not downloaded in $attempts attempts of ${attempt_minutes} min each (slow or failing apt mirror)"
      exit 1
    fi
  done
  sudo apt-get "${apt_opts[@]}" install -y --no-download ffmpeg
  stage_installed_set
  report "cache miss"
}

# В кэш идут только .deb тех версий, что стоят после установки: архив прерванной попытки, который
# сменила новая версия с зеркала, в набор не попадает. Набор без единственного ffmpeg_*.deb не
# сохраняется: путь попадания такой набор отвергнет.
stage_installed_set() {
  shopt -s nullglob
  local deb meta installed left_out=0
  mkdir -p "$debs"
  for deb in "$archives"/*.deb; do
    meta=$(dpkg-deb --show --showformat='${Package} ${Version}' "$deb") || meta=""
    installed=""
    if [ -n "$meta" ]; then
      installed=$(dpkg-query -W -f='${db:Status-Status} ${Version}\n' "${meta%% *}" 2>/dev/null) ||
        installed=""
    fi
    if [ -n "$meta" ] && grep -qxF "installed ${meta#* }" <<<"$installed"; then
      cp "$deb" "$debs"/
    else
      left_out=$((left_out + 1))
    fi
  done
  local files=("$debs"/*.deb) packages=("$debs"/ffmpeg_*.deb)
  if [ "$left_out" -gt 0 ]; then
    echo "$left_out downloaded .deb files are not of the installed package set and stay out of the ffmpeg package cache"
  fi
  if [ "${#packages[@]}" -ne 1 ]; then
    rm -rf "$debs"
    echo "::warning::the downloaded set has no single ffmpeg_*.deb (${#files[@]} files): nothing goes to the ffmpeg package cache"
    return
  fi
  echo "${#files[@]} downloaded .deb files go to the ffmpeg package cache"
}

case "${1:-}" in
  key) key ;;
  install)
    if [ "${CACHE_HIT:-}" = "true" ]; then install_from_cache; else download_and_install; fi
    ;;
  *)
    echo "usage: install.sh key|install" >&2
    exit 2
    ;;
esac
