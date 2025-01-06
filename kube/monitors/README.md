# Viewing Results

Best way to do this is hands down port-forwarding.  

1. Launch the results monitor with the following:
```
kubectl apply -f /path/to/nfe-monitor.yaml
```
2. Exec into the pod once its running:
```
kubectl exec -it ethan-nfe-monitor -- /bin/bash
```
3. Optionally navigate to desired subdirectory /develop/results/ etc etc
4. Start an http server in the pod:
```
python3 -m http.server 8080
```
5. Open a new terminal window and forward the pod port to your local machine:
```
kubectl port-forward ethan-nfe-monitor 8080:8080
```
6. Access the files in a browser:
```
http://localhost:8080
```