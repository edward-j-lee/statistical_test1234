docker stop $(docker ps -q --filter ancestor="stattest")
git pull
docker build -t stattest .
docker run --restart=always -p 8000:8000 stattest