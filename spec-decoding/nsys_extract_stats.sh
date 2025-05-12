for file in spec-decoding/llama3/nsys/buggy/llama3_*.nsys-rep; do
    base=${file%.nsys-rep}
    echo "📊 Processing $file..."
    nsys stats "$file" > "$base.txt"
done
