#!/usr/bin/env bash
# Установка ffmpeg в CI: один путь для jobs checks и clean-install (#183).
#   install.sh key      — ключ кэша набора .deb и restore-keys в $GITHUB_OUTPUT.
#   install.sh install  — набор из $FFMPEG_DEBS через dpkg -i, без apt-get и без зеркала apt:
#                         CACHE_HIT=true — набор этого образа, RESTORED_KEY — набор другого образа
#                         той же ОС, архитектуры и уровня CPU (#184); ключ другого уровня — промах.
#                         Набор другого образа, который здесь не встал, снимается, и установка идёт
#                         с зеркала. Без набора — скачивание с ограниченными
#                         повторами (#65, #70), установка и копия .deb установленных версий в
#                         $FFMPEG_DEBS для actions/cache/save.
set -euo pipefail

debs=${FFMPEG_DEBS:?FFMPEG_DEBS is not set}
archives=${APT_ARCHIVES:-/var/cache/apt/archives}

# Точный ключ — образ runner'а: у каждого образа свой набор. restore-keys без версии образа: runner
# на образе без своего набора берёт набор другого образа той же ОС, архитектуры и уровня CPU (#184),
# а install_set проверяет, что он здесь годится. Второй префикс берёт и набор ключа #183 (версия
# образа там идёт до архитектуры), пока набора нового ключа нет. Префикс сам уровень не проверяет:
# ключ другой архитектуры или уровня CPU отвергает restored_key_fits.
cpu_tag() {
  local level
  # apt берёт сборки под уровень CPU (amd64v3): набор с одного хоста не годится хосту ниже уровнем.
  level=$(/lib64/ld-linux-x86-64.so.2 --help 2>/dev/null |
    sed -n 's/^ *\(x86-64-v[0-9]\) (supported.*/\1/p' | sed -n 1p) || true
  echo "$(dpkg --print-architecture)-${level:-cpu-unknown}"
}

key() {
  local script_sha cpu
  script_sha=$(sha256sum "${BASH_SOURCE[0]}" | cut -c1-16)
  cpu=$(cpu_tag)
  if [ -n "${ImageOS:-}" ] && [ -n "${ImageVersion:-}" ]; then
    {
      echo "key=ffmpeg-debs-$ImageOS-$cpu-$ImageVersion-$script_sha"
      printf 'restore-keys<<EOF\n%s\n%s\nEOF\n' "ffmpeg-debs-$ImageOS-$cpu-" "ffmpeg-debs-$ImageOS-"
    } >>"$GITHUB_OUTPUT"
  else
    echo "::warning::ImageOS or ImageVersion is not set: the ffmpeg package cache serves this run only"
    {
      echo "key=ffmpeg-debs-unknown-image-run-${GITHUB_RUN_ID:-0}-$cpu-$script_sha"
      echo "restore-keys="
    } >>"$GITHUB_OUTPUT"
  fi
}

# Ключ нового вида ffmpeg-debs-<ОС>-<cpu>-<образ>-<sha> или ключ #183 ffmpeg-debs-<ОС>-<образ>-<cpu>-<sha>
# с архитектурой и уровнем CPU этого хоста.
restored_key_fits() {
  local cpu rest
  [ -n "${ImageOS:-}" ] || return 1
  cpu=$(cpu_tag)
  case "$RESTORED_KEY" in
    "ffmpeg-debs-$ImageOS-$cpu-"*) return 0 ;;
    "ffmpeg-debs-$ImageOS-"*) rest=${RESTORED_KEY#"ffmpeg-debs-$ImageOS-"} ;;
    *) return 1 ;;
  esac
  case "$rest" in
    -*) return 1 ;;
    *-*) [[ ${rest#*-} == "$cpu-"* ]] ;;
    *) return 1 ;;
  esac
}

report() {
  local version
  version=$(ffmpeg -version)
  echo "${version%%$'\n'*}"
  echo "ffmpeg installed in ${SECONDS} s ($1)"
}

# Ставит набор из $debs ($1 — exact или other). Набор этого образа (exact) ставится целиком: в нём и
# обновления пакетов образа, которые взяла исходная установка с зеркала. Набор другого образа (other)
# пакеты образа не меняет: .deb пакета, который на образе уже стоит (любой версии), не ставится, а
# зависимости остальных проверяет dpkg -i. Набор годится, когда ffmpeg затем кодирует тестовый
# сигнал. Негодный набор снимается dpkg --purge — только пакеты, которых на образе не было. Причина
# отказа — в $unfit.
install_set() {
  local mode=$1
  shift
  shopt -s nullglob
  local files=("$debs"/*.deb) packages=("$debs"/ffmpeg_*.deb) deb meta status
  local picked=() names=() kept=0
  unfit=""
  if [ "${#packages[@]}" -ne 1 ]; then
    unfit="the ffmpeg package cache has no single ffmpeg_*.deb (${#files[@]} files)"
    return 1
  fi
  for deb in "${files[@]}"; do
    if ! meta=$(dpkg-deb --show --showformat='${Package} ${Version}' "$deb"); then
      unfit="the cached $(basename "$deb") is not a readable .deb"
      return 1
    fi
    status=$(dpkg-query -W -f='${db:Status-Status} ${Version}\n' "${meta%% *}" 2>/dev/null) ||
      status=""
    if [ "${status%% *}" = installed ]; then
      if [ "$mode" = other ]; then
        kept=$((kept + 1))
      else
        picked+=("$deb")
      fi
    else
      picked+=("$deb")
      names+=("${meta%% *}")
    fi
  done
  echo "$1: dpkg -i of ${#picked[@]} cached .deb files ($(basename "${packages[0]}")); no apt-get update, nothing downloaded from the apt mirror"
  if [ "$kept" -gt 0 ]; then
    echo "$kept cached .deb files are of packages the runner image already has: the image keeps its own"
  fi
  if [ "${#picked[@]}" -gt 0 ] && ! sudo dpkg -i "${picked[@]}"; then
    unfit="the cached ffmpeg packages did not install with dpkg -i"
  elif ! ffmpeg -nostdin -hide_banner -loglevel error -f lavfi -i sine=duration=0.1 -f null -; then
    unfit="ffmpeg of the cached packages does not encode a test signal"
  else
    return 0
  fi
  if [ "${#names[@]}" -gt 0 ] &&
    ! sudo dpkg --purge --force-depends --force-remove-reinstreq "${names[@]}"; then
    echo "::warning::dpkg --purge did not remove the cached ffmpeg packages"
  fi
  return 1
}

install_from_cache() {
  if ! install_set exact "ffmpeg package cache hit"; then
    echo "::error::$unfit; delete the cache entry with gh cache delete and rerun"
    exit 1
  fi
  report "cache hit"
}

install_restored() {
  if install_set other "ffmpeg package set of another runner image ($RESTORED_KEY)"; then
    report "set of another runner image"
  else
    echo "::warning::$unfit: the set of another runner image does not fit this one, download from the apt mirror"
    download_and_install
  fi
}

download_and_install() {
  echo "ffmpeg package cache miss: download from the apt mirror with bounded retry"
  apt_opts=(-q -o Acquire::Retries=3 -o Acquire::http::Timeout=30)
  attempts=3
  attempt_minutes=5
  # Набор для кэша — ровно то, что скачает эта установка. .deb, скачанные попыткой, которую оборвал
  # timeout, остаются в кэше apt, и следующая попытка продолжает с них.
  sudo apt-get "${apt_opts[@]}" clean
  # .deb набора другого образа, что здесь не встал: apt берёт те, чья сумма совпала с индексом
  # зеркала, и докачивает только остальное.
  if [ -d "$debs" ]; then
    shopt -s nullglob
    local seed=("$debs"/*.deb)
    if [ "${#seed[@]}" -gt 0 ]; then
      sudo cp "${seed[@]}" "$archives"/
    fi
    rm -rf "$debs"
  fi
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
    if [ "${CACHE_HIT:-}" = "true" ]; then
      install_from_cache
    elif [ -n "${RESTORED_KEY:-}" ] && [ -d "$debs" ]; then
      if restored_key_fits; then
        install_restored
      else
        echo "::warning::the restored ffmpeg package set ($RESTORED_KEY) is not of this OS, arch and CPU level ($(cpu_tag)): download from the apt mirror"
        rm -rf "$debs"
        download_and_install
      fi
    else
      download_and_install
    fi
    ;;
  *)
    echo "usage: install.sh key|install" >&2
    exit 2
    ;;
esac
