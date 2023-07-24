-------------- scheduleCompute(A)
upload A            
                    wait for upload A
                    compute A
                    A done
-------------- scheduleCompute(B)
upload B            
                    wait for upload B
                    compute B
                    B done
-------------- checkResult(A)
wait for A done
download A 
-------------- scheduleCompute(A)
upload A
                    wait for upload A
                    compute A
                    A done
--------------- checkResult(B)
wait for B done
download B
--------------- scheduleCompute(B)
upload B
                    wait for upload B
                    compute B
                    B done
--------------- checkResult(A)
wait for A done
download A 
--------------- scheduleCompute(A)
upload A
                    wait for upload A
                    compute A
                    A done
....

