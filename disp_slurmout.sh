for file in "$@"
do 
	echo -n $file"   "; 
	cat $file | grep "Iter" | tail -n 1; 
done
