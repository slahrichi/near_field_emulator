# Viewing Results

Best way to do this is hands down port-forwarding.  

1. Launch the results monitor with the following:
```
kubectl apply -f /path/to/monitor-results.yaml
```
2. Exec into the pod once its running:
```
kubectl exec -it ethan-monitor-results -- /bin/bash
```
3. Optionally navigate to desired subdirectory /develop/results/ etc etc
4. Start an http server in the pod:
```
python3 -m http.server 8080
```
5. Forward the pod port to your local machine
```
kubectl port-forward ethan-monitor-results 8080:8080
```
6. Access the files in a browser:
```
http://localhost:8080
```