device=2

for model in "Pyraformer" "Autoformer" "Informer" "Reformer" "MICN" "Crossformer" "FiLM" "SCINet" "PAttn" "FreTS"
    do 
        python -u run.py --task_name classification --is_training 1 --root_path ./dataset/17/ --model_id Heartbeat --model $model --data PDM --e_layers 3 --batch_size 4 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10 --gpu $device
    done
done