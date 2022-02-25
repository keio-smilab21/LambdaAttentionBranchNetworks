# for seed in 42; do
#     python visualize.py -c checkpoints/base_seed${seed}/config.json --method RISE --insdel_step 5 --block_size 16
# done

# for seed in 38 39 40 41; do
#     python visualize.py -c checkpoints/base_seed${seed}/config.json --insdel_step 500 --method npy --theta_att 0 --block_size 1 --attention_dir outputs/RISE16_evalsame_base_seed${seed}
#     python visualize.py -c checkpoints/base_seed${seed}/config.json --insdel_step 100 --method npy --theta_att 0 --block_size 2 --attention_dir outputs/RISE16_evalsame_base_seed${seed}
#     python visualize.py -c checkpoints/base_seed${seed}/config.json --insdel_step 20 --method npy --theta_att 0 --block_size 4 --attention_dir outputs/RISE16_evalsame_base_seed${seed}
#     python visualize.py -c checkpoints/base_seed${seed}/config.json --insdel_step 10 --method npy --theta_att 0 --block_size 8 --attention_dir outputs/RISE16_evalsame_base_seed${seed}
#     python visualize.py -c checkpoints/base_seed${seed}/config.json --insdel_step 5 --method npy --theta_att 0 --block_size 16 --attention_dir outputs/RISE16_evalsame_base_seed${seed}
# done


# for seed in 38 39 40 41 42; do
#     python visualize.py -c checkpoints/finetune_theta50_seed${seed}/config.json --insdel_step 500 --block_size 1
#     python visualize.py -c checkpoints/finetune_theta50_seed${seed}/config.json --insdel_step 100 --block_size 2
#     python visualize.py -c checkpoints/finetune_theta50_seed${seed}/config.json --insdel_step 20 --block_size 4
#     python visualize.py -c checkpoints/finetune_theta50_seed${seed}/config.json --insdel_step 10 --block_size 8
#     python visualize.py -c checkpoints/finetune_theta50_seed${seed}/config.json --insdel_step 5 --block_size 16
# done

# for seed in 38 39 40 41 42; do
#     python train.py -c configs/idrid_lambda_abn_vis.json --seed ${seed} --base_pretrained2 checkpoints/base_${seed}/best.py --theta_att 0.25 --run_name id_finetune_theta25_w5_seed${seed} --freeze fe --loss_weights 5
    # python visualize.py -c checkpoints/id_finetune_theta25_w5_seed${seed}/config.json --insdel_step 500 --block_size 1
    # python visualize.py -c checkpoints/id_finetune_theta25_w5_seed${seed}/config.json --insdel_step 100 --block_size 2
    # python visualize.py -c checkpoints/id_finetune_theta25_w5_seed${seed}/config.json --insdel_step 20 --block_size 4
    # python visualize.py -c checkpoints/id_finetune_theta25_w5_seed${seed}/config.json --insdel_step 10 --block_size 8
    # python visualize.py -c checkpoints/id_finetune_theta25_w5_seed${seed}/config.json --insdel_step 5 --block_size 16
# done

# for seed in 41 42; do
#     python train.py -c configs/idrid_lambda_abn_vis.json --seed ${seed} --base_pretrained2 checkpoints/base_${seed}/best.py --theta_att 0.5 --run_name finetune_theta50_seed${seed} --freeze fe --loss_weights 5
# done

# for seed in 38 39 40 41 42; do
#     python train.py -c configs/idrid_abn.json --loss_weights 2 1 --lambda_att 1 --lambda_var 0 --lr_ab 1e-3 --seed $seed --batch_size 32 --theta_att 0 --run_name abn_cv_seed${seed}


# done



# for seed in 38 39 40 41 42; do
#     runname=idridlambda_th0_cv_seed${seed};
#     python train.py -c configs/idrid_lambda_abn_vis.json --loss_weights 2 1 --lambda_att 1 --lambda_var 0 --lr_ab 1e-3 --seed $seed --batch_size 32 --theta_att 0 --run_name ${runname}
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 500 --block_size 1
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 100 --block_size 2
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 20 --block_size 4
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 10 --block_size 8
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 5 --block_size 16
# done

# for seed in 41 42; do
#     runname=base_RE_seed${seed}
#     att_dir=outputs/erroranalysis_RISE16_evalsame_base_RE_seed${seed}
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 500 --block_size 1 --method npy --attention_dir ${att_dir}
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 100 --block_size 2 --method npy --attention_dir ${att_dir}
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 20 --block_size 4 --method npy --attention_dir ${att_dir}
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 10 --block_size 8 --method npy --attention_dir ${att_dir}
#     python visualize.py -c checkpoints/${runname}/config.json --insdel_step 5 --block_size 16 --method npy --attention_dir ${att_dir}
# done

# python visualize.py -c checkpoints/base_noRE_seed38/config.json --block_size 2  --insdel_step 100 --method lambda
# python visualize.py -c checkpoints/base_noRE_seed38/config.json --block_size 1  --insdel_step 500 --method lambda

run_name=RE0.75
cfgf=checkpoints/${run_name}/config.json
python visualize.py -c ${cfgf} --block_size 16  --insdel_step 5
python visualize.py -c ${cfgf} --block_size 8  --insdel_step 10
python visualize.py -c ${cfgf} --block_size 4  --insdel_step 20
python visualize.py -c ${cfgf} --block_size 2  --insdel_step 100
python visualize.py -c ${cfgf} --block_size 1  --insdel_step 500

for seed in 38 39 40 41; do
    run_name=RE0.75${seed}
    cfgf=checkpoints/${run_name}/config.json
    python visualize.py -c ${cfgf} --block_size 16  --insdel_step 5
    python visualize.py -c ${cfgf} --block_size 8  --insdel_step 10
    python visualize.py -c ${cfgf} --block_size 4  --insdel_step 20
    python visualize.py -c ${cfgf} --block_size 2  --insdel_step 100
    python visualize.py -c ${cfgf} --block_size 1  --insdel_step 500
done


# done

# for div in layer1 layer3; do
#     for seed in 38 39 40 41 42; do
#         runname=idridlambda_layer${div}_0.75_cv_seed${seed};
#         python visualize.py -c checkpoints/${runname}/config.json --insdel_step 500 --block_size 1
#         python visualize.py -c checkpoints/${runname}/config.json --insdel_step 100 --block_size 2
#         python visualize.py -c checkpoints/${runname}/config.json --insdel_step 20 --block_size 4
#         python visualize.py -c checkpoints/${runname}/config.json --insdel_step 10 --block_size 8
#         python visualize.py -c checkpoints/${runname}/config.json --insdel_step 5 --block_size 16
#     done
# done

# for seed in 38 39 40 41 42; do
#     python train.py -c configs/idrid_lambda_abn_vis.json --loss_weights 2 1 --lambda_att 3 --lambda_var 0 --lr_ab 1e-3 --seed $seed --batch_size 32 --theta_att 0.75 --run_name cv_seed${seed}
# done

# for seed in 38 39 40 41 42; do
#     python train.py -c configs/idrid_lambda_abn_vis.json --loss_weights 2 1 --lambda_att 3 --lambda_var 0 --lr_ab 1e-3 --seed $seed --batch_size 32 --div layer3
# done

# for f in checkpoints/idrid_lambda_abn_vis_divlayer*_2021-12-2*_1*; do
# python visualize.py -c ${f}/config.json
# done

# for f in checkpoints/idrid_lambda_abn_vis_divlayer*_2021-12-23*; do
# python visualize.py -c ${f}/config.json
# done