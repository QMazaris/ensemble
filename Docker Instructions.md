## Already done
echo YOUR_PAT_HERE | docker login ghcr.io -u qmazaris --password-stdin

## Must be done to send the new images up
```bash
docker build -t ensemble-backend --target backend .
docker build -t ensemble-frontend --target frontend .
```
```bash
docker tag ensemble-backend ghcr.io/qmazaris/ensemble-backend:latest
docker tag ensemble-frontend ghcr.io/qmazaris/ensemble-frontend:latest
```
```bash
docker push ghcr.io/qmazaris/ensemble-backend:latest
docker push ghcr.io/qmazaris/ensemble-frontend:latest
```

## If the images need to be updated, first you must delete them
```bash
docker rmi -f ghcr.io/qmazaris/ensemble-backend:latest
docker rmi -f ghcr.io/qmazaris/ensemble-frontend:latest
```
