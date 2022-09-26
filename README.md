# Deep-Backdoor-Attack

## Pipeline
![Pipeline](./images/pipeline.png)
## Instruction
Step 1: Trigger Generation
```shell
python trigger_generation.py --cuda 1
```
Step 2: Backdoor Injection
```shell
python backdoor_train.py --cuda 1 --trigger_type "dba"
```

## Results

