1. Open this repo using DevContainers in VS code

2. To develop,
        Run:
        ```
        export PYTHONPATH=$(pwd):$PYTHONPATH
        ```
         run the following in the `nerfstudio/hdr_nerf` folder.
        ```
        pip install -e .
        ```

3. To train
    ```
    ns-train hdr-nerf --data data/nerfstudio/chair
    ```

