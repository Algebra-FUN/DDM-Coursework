mkdir $1s
cd $1s

output_file="time till now.txt"

for i in $(seq 1 $2)
do
dir_name=$1$i
mkdir $dir_name
cd $dir_name
nanotimestamp=$(date +%s%N)
echo "microseconds since 1970-01-01 00:00:00 UTC:" > "$output_file"
echo "<$((nanotimestamp/1000))>" >> "$output_file"
cd ..
done