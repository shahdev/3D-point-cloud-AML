#!/bin/bash

mkdir dag_inp_0.01
python evaluate_attack.py --group=0 --arch=original --load=orig-ft_it22000 --alpha_inp=0.01 --alpha_flow=0 --tau=0 --spatial_dag=0 --save_dir dag_inp_0.01 > dag_inp_0.01/out.txt

mkdir dag_inp_0.05
python evaluate_attack.py --group=0 --arch=original --load=orig-ft_it22000 --alpha_inp=0.05 --alpha_flow=0 --tau=0 --spatial_dag=0 --save_dir dag_inp_0.05 > dag_inp_0.05/out.txt


mkdir dag_inp_0.1
python evaluate_attack.py --group=0 --arch=original --load=orig-ft_it22000 --alpha_inp=0.1 --alpha_flow=0 --tau=0 --spatial_dag=0 --save_dir dag_inp_0.1 > dag_inp_0.1/out.txt


mkdir dag_inp_0.5
python evaluate_attack.py --group=0 --arch=original --load=orig-ft_it22000 --alpha_inp=0.5 --alpha_flow=0 --tau=0 --spatial_dag=0 --save_dir dag_inp_0.5 > dag_inp_0.5/out.txt


mkdir dag_inp_1
python evaluate_attack.py --group=0 --arch=original --load=orig-ft_it22000 --alpha_inp=1 --alpha_flow=0 --tau=0 --spatial_dag=0 --save_dir dag_inp_1 > dag_inp_1/out.txt

mkdir dag_inp_5
python evaluate_attack.py --group=0 --arch=original --load=orig-ft_it22000 --alpha_inp=5 --alpha_flow=0 --tau=0 --spatial_dag=0 --save_dir dag_inp_5 > dag_inp_5/out.txt


mkdir dag_inp_10
python evaluate_attack.py --group=0 --arch=original --load=orig-ft_it22000 --alpha_inp=10 --alpha_flow=0 --tau=0 --spatial_dag=0 --save_dir dag_inp_10 > dag_inp_10/out.txt

