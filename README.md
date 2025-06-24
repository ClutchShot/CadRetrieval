# CadRetrieval

## Training
```
    python contrast.py train --dataset FABWave --dataset_path 
    ./data/FABWave --max_epochs 10 --batch_size 64 
    --num_workers 0 --gpus 1 --auto_select_gpus true
```


## Evaluation
```
    python evaluation.py --dataset FABWave --dataset_path 
    ./data/FABWave --max_epochs 10 --batch_size 64 
    --num_workers 0 --gpus 1 --auto_select_gpus true 
    --checkpoint ./results/0619/133643/best.ckpt --db_path ./data/vec_db_test

```