export N=10
rm logs/agent*

for (( c=0; c<$N; c++))
do
   let a=30000+$c
   fuser -k $a/tcp
done

# for (( c=0; c<$N; c++))
# do
#    python ./agent.py -i $c -n $N& 
# done


