#!/bin/sh
set -eu

docker run --rm --network host \
  -v /opt/1panel/www/sites/metatrans.aicarb.com/index:/var/www/html \
  -v /opt/1panel/www/sites/metatrans.aicarb.com/ssl:/etc/letsencrypt \
  certbot/certbot:latest renew --webroot -w /var/www/html --quiet

docker exec 1Panel-openresty-SP78 openresty -t
docker exec 1Panel-openresty-SP78 openresty -s reload
