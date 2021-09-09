docker stop `docker ps -q -a`
git pull
docker build -t stattest .
docker run --restart=always -p 8000:8000 stattest