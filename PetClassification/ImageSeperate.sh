for f in *.jpg           # no need to use ls.
do
    filename=${f##*/}          # Use the last part of a path.
    dir=${filename%.*}    # Remove from the last dot.
    dir=${dir%_*}              # Remove all dots.
    
    #echo "$filename $dir"     # Test naming output
    if [[ -d $dir ]]; then     # If the directory exists
        mv "$filename" "$dir"/ # Move file there.
    else
        mkdir -vp "$dir"
        mv "$filename" "$dir"
    fi

done
