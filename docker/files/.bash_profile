
run_tests(){
    python -m unittest coverage discover -s /home/code/tests
}

run_jupyter_serv(){
    /usr/local/bin/jupyter notebook \
     --ip 0.0.0.0 \
     --port 8888 \
     --no-browser \
     --allow-root \
     --NotebookApp.allow_origin_pat=https://.*vscode-cdn\\.net \
     --notebook-dir /tf/

}