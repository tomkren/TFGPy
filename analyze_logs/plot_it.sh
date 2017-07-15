#!/usr/bin/env bash

sed_hell () {
    sed -n 's/FitnessCache<\([0-9]*\)>[ @]*\(AT END\)\{,1\}[[:space:]]*\([0-9]*\)[[:space:]]*max_fitness=\([^[:space:]]*\)[[:space:]].*$/\1\t\3\t\4\t\2/p'
}

anal () {
    python3 analyze.py "$@"
}

# parity

par_apptree_lv2 () {
cat $DIR/u-pl12/LOG_EXPERIMENT_parity_30_nmcs_level_2
}

par_apptree_lv3 () {
cat $DIR/u-pl12/LOG_EXPERIMENT_parity_30_nmcs_level_3
}

par_stack_lv2 () {
cat $DIR/u-pl13/LOG_EXPERIMENT_parity_30_nmcs_level_2
}

par_stack_lv3 () {
cat $DIR/u-pl13/LOG_EXPERIMENT_parity_30_nmcs_level_3
}

# physics

ph_smart_lv2 () {
cat $DIR/u-pl8/LOG_EXPERIMENT_physics_smart_31_nmcs_level_2 $DIR/u-pl1/LOG_TMP_physics_smart_31_nmcs_level_2
}

ph_dumb_lv2 () {
cat haf/*
}

ph_smart_lv1 () {
cat $DIR/u-pl2/LOG_upl2_1000xlevel1_physics_smart_31_nmcs_level_1 $DIR/u-pl7/LOG_EXPERIMENT_physics_smart_31_nmcs_level_1
}

ph_smart_mcts () {
cat $DIR/u-pl3/LOG_EXPERIMENT_physics_smart_31_mcts  $DIR/u-pl5/LOG_EXPERIMENT_physics_smart_31_nmcs_level_1
}

ph_smart_mcts_08 () {
cat $DIR/u-pl4/LOG_EXPERIMENT_physics_smart_31_mcts $DIR/u-pl9/LOG_EXPERIMENT_physics_smart_31_mcts
}

ph_smart_mcts_avg () {
cat $DIR/u-pl11/LOG_EXPERIMENT_physics_smart_31_mcts $DIR/u-pl10/LOG_EXPERIMENT_physics_smart_31_mcts
}

DIR=1000

par_apptree_lv2 | sed_hell | anal > DATA/par_apptree_lv2
par_apptree_lv3 | sed_hell | anal > DATA/par_apptree_lv3
par_stack_lv2 | sed_hell | anal > DATA/par_stack_lv2
par_stack_lv3 | sed_hell | anal > DATA/par_stack_lv3

ph_smart_lv2  | sed_hell | anal  > DATA/ph_smart_lv2
ph_dumb_lv2  | sed_hell | anal  > DATA/ph_dumb_lv2

ph_smart_lv1 | sed_hell | anal  > DATA/ph_smart_lv1
ph_smart_mcts | sed_hell | anal  > DATA/ph_smart_mcts
ph_smart_mcts_08 | sed_hell | anal  > DATA/ph_smart_mcts_08
ph_smart_mcts_avg | sed_hell | anal  > DATA_ph/smart_mcts_avg

