python test.py \
--name c_styleD_E12d0_outN_shortepoch_3T3_4 \
--A_file ./datasets/cell_test_A.txt \
--M_file ./datasets/cell_test_N.txt  \
--N_file ./datasets/cell_test_M.txt \
--model tcell \
--netT original \
--num_res_blocks 2 \
--n_encoders 12 \
--n_decoders 0 \
--netD style \
--gpu_ids 0 \
--load_size 384 \
--fine_size 384 \
--batch_size 1 \
--attn_G \
--which_iter 0
#modify 2-5
#line 17 you can change based on iteration
#--which_iter 0 means it tests for the final epoch
#batch_size=1 for all testing