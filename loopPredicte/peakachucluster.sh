#!/bin/bash
resolutions=("5000")
input_files=(
             "/Path/to/chaname.bedpe"
)
thres=("0.9")

for res in "${resolutions[@]}"; do
    for thre in "${thres[@]}"; do
        output_files=()
        for input_file in "${input_files[@]}"; do
            file_name="${input_file##*/}"
            file_name_no_extension="${file_name%.*}"    
            output_file="${input_file%/*}/${file_name_no_extension}_loop${thre}.bedpe"
    
            peakachu pool -r "$res" -i "$input_file" -o "$output_file" -t ${thre}

            output_files+=("$output_file")
        done

        output_dir="${input_files[0]%/*}"
        merged_output_file="${output_dir}/sum_chr_loopcluster${thre}.bedpe"
        cat "${output_files[@]}" > "$merged_output_file"
        echo "Merged output saved to $merged_output_file"

        num_lines=$(wc -l < "$merged_output_file")
        echo "Number of lines in merged output file ${thre}: $num_lines"

        renamed_output_file="${output_dir}/sum${num_lines}_chr_default_loop${thre}.bedpe"
        mv "$merged_output_file" "$renamed_output_file"
        for output_file in "${output_files[@]}"; do
            rm "$output_file"
            echo "Deleted $output_file"
        done
    done
done

echo "All processing completed."

