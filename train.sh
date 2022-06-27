python train.py \
--name c_styleD_E12d0_outN_shortepoch_3T3_4 \
--A_file /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25572/dataset/image_painting/image_list/cell_A.txt \
--M_file /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25572/dataset/image_painting/image_list/cell_A.txt \
--N_file /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25572/dataset/image_painting/image_list/cell_A.txt \
--model tcell \
--netT original \
--num_res_blocks 2 \
--n_encoders 12 \
--n_decoders 0 \
--netD style \
--gpu_ids 0 \
--lambda_g 1 \
--load_size 464 \
--fine_size 384 \
--batch_size 1 \
--display_port 8093 \
--attn_G \
--continue_train
#modify line 2-5