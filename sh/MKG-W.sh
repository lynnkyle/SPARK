nohup python Trainer.py \
  --data MKG-W \
  --lr 5e-4 \
  --dim 256 \
  --num_epoch 3000 \
  --valid_epoch 10 \
  --log_epoch 100 \
  --exp SPARK_MKGW_16_16_k3 \
  --num_layer_enc_ent 1 \
  --num_layer_enc_rel 1 \
  --num_layer_dec 2 \
  --num_head 4 \
  --hidden_dim 1024 \
  --dropout 0.01 \
  --emb_dropout 0.9 \
  --vis_dropout 0.4 \
  --txt_dropout 0.1 \
  --smoothing 0.0 \
  --batch_size 2048 \
  --decay 0.0 \
  --step_size 50 \
  --max_vis_token 16 \
  --max_txt_token 16 \
  --neg_num 3 \
  --margin 0.1 \
  --fusion_function ssm \
  --score_function tucker \
  --loss_modality 0.5 \
  --loss_entity 10 \
  > SPARK_MKGW_16_16_k3.log 2>&1 &

#超参数实验 base_entity_k （5）
CUDA_VISIBLE_DEVICES=1
nohup python Trainer.py \
  --data MKG-W \
  --lr 5e-4 \
  --dim 256 \
  --num_epoch 3000 \
  --valid_epoch 10 \
  --log_epoch 100 \
  --exp SPARK_MKGW_0_0_k3 \
  --num_layer_enc_ent 1 \
  --num_layer_enc_rel 1 \
  --num_layer_dec 2 \
  --num_head 4 \
  --hidden_dim 1024 \
  --dropout 0.01 \
  --emb_dropout 0.9 \
  --vis_dropout 0.4 \
  --txt_dropout 0.1 \
  --smoothing 0.0 \
  --batch_size 2048 \
  --decay 0.0 \
  --step_size 50 \
  --max_vis_token 0 \
  --max_txt_token 0 \
  --neg_num 3 \
  --margin 0.1 \
  --fusion_function ssm \
  --score_function tucker \
  > SPARK_MKGW_0_0_k3.log 2>&1 &

nohup python Trainer.py \
  --data MKG-W \
  --lr 5e-4 \
  --dim 256 \
  --num_epoch 3000 \
  --valid_epoch 10 \
  --log_epoch 100 \
  --exp SPARK_MKGW_16_32_k5 \
  --num_layer_enc_ent 1 \
  --num_layer_enc_rel 1 \
  --num_layer_dec 2 \
  --num_head 4 \
  --hidden_dim 1024 \
  --dropout 0.01 \
  --emb_dropout 0.9 \
  --vis_dropout 0.4 \
  --txt_dropout 0.1 \
  --smoothing 0.0 \
  --batch_size 2048 \
  --decay 0.0 \
  --step_size 50 \
  --max_vis_token 16 \
  --max_txt_token 32 \
  --neg_num 5 \
  --margin 0.1 \
  --fusion_function ssm \
  --score_function tucker \
  > SPARK_MKGW_16_32_k5.log 2>&1 &

nohup python Trainer.py \
  --data MKG-W \
  --lr 5e-4 \
  --dim 256 \
  --num_epoch 3000 \
  --valid_epoch 10 \
  --log_epoch 100 \
  --exp SPARK_MKGW_16_32_k7 \
  --num_layer_enc_ent 1 \
  --num_layer_enc_rel 1 \
  --num_layer_dec 2 \
  --num_head 4 \
  --hidden_dim 1024 \
  --dropout 0.01 \
  --emb_dropout 0.9 \
  --vis_dropout 0.4 \
  --txt_dropout 0.1 \
  --smoothing 0.0 \
  --batch_size 2048 \
  --decay 0.0 \
  --step_size 50 \
  --max_vis_token 16 \
  --max_txt_token 32 \
  --neg_num 7 \
  --margin 0.1 \
  --fusion_function ssm \
  --score_function tucker \
  > SPARK_MKGW_16_32_k7.log 2>&1 &

nohup python Trainer.py \
  --data MKG-W \
  --lr 5e-4 \
  --dim 256 \
  --num_epoch 3000 \
  --valid_epoch 10 \
  --log_epoch 100 \
  --exp SPARK_MKGW_16_32_k9 \
  --num_layer_enc_ent 1 \
  --num_layer_enc_rel 1 \
  --num_layer_dec 2 \
  --num_head 4 \
  --hidden_dim 1024 \
  --dropout 0.01 \
  --emb_dropout 0.9 \
  --vis_dropout 0.4 \
  --txt_dropout 0.1 \
  --smoothing 0.0 \
  --batch_size 2048 \
  --decay 0.0 \
  --step_size 50 \
  --max_vis_token 16 \
  --max_txt_token 32 \
  --neg_num 9 \
  --margin 0.1 \
  --fusion_function ssm \
  --score_function tucker \
  > SPARK_MKGW_16_32_k9.log 2>&1 &

# mod / ent
nohup python Trainer.py \
  --data MKG-W \
  --lr 5e-4 \
  --dim 256 \
  --num_epoch 3000 \
  --valid_epoch 10 \
  --log_epoch 100 \
  --exp SPARK_MKGW_0.05_1 \
  --num_layer_enc_ent 1 \
  --num_layer_enc_rel 1 \
  --num_layer_dec 2 \
  --num_head 4 \
  --hidden_dim 1024 \
  --dropout 0.01 \
  --emb_dropout 0.9 \
  --vis_dropout 0.4 \
  --txt_dropout 0.1 \
  --smoothing 0.0 \
  --batch_size 2048 \
  --decay 0.0 \
  --step_size 50 \
  --max_vis_token 16 \
  --max_txt_token 16 \
  --neg_num 3 \
  --margin 0.1 \
  --fusion_function ssm \
  --score_function tucker \
  --loss_modality 0.05 \
  --loss_entity 1 \
  > SPARK_MKGW_0.05_1.log 2>&1 &