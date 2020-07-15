#!/bin/bash

function blockUntilBertServerIsRunning {
        nc -z localhost 5555
        while [ "$?" -ne "0" ] 
        do
                echo "BERT-Server is not running. sleep 10s..."
                sleep 10s
                nc -z localhost 5555
        done
}

echo "Restore Graphs..."
python3 restore_tf_graph.py --bert_dir './uncased_L-12_H-768_A-12/' > $2/python-3-restore.logs 2>&1
python restore_tf_graph.py --bert_dir './uncased_L-12_H-768_A-12/' > $2/python-restore.logs 2>&1
echo "graph restore done"

/home/Katana/.local/bin/bert-serving-start -model_dir './uncased_L-12_H-768_A-12/' > $2/bert-serving.logs 2>&1 &

blockUntilBertServerIsRunning

echo "service start done"
echo $1
echo $2

sleep 30s

python3 task_tira_3.py --ii $1 --o $2
echo "run"
