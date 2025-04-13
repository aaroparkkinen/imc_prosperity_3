input_filepath = "/Users/urthor/projects/imc_prosperity/data/logs-round-1/round_1.log"
output_filepath = "/Users/urthor/projects/imc_prosperity/data/bottle-round-1-island-data/prices_round_1_day_1.csv"

def extract_and_overwrite(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    start_marker = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
    start_index = None
    end_index = None

    # Find the start index
    for i, line in enumerate(lines):
        # Print if the line starts with "day"
        if lines[i][0] == "d":
            print(lines[i])
            print(i)
        if line.strip() == start_marker:
            start_index = i
            break

    print("Start index is " + str(start_index))

    # Find the end index
    if start_index is not None:
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip() == "":
                end_index = i
                break

    # Extract the relevant section
    if start_index is not None and end_index is not None:
        extracted_lines = lines[start_index:end_index]
    elif start_index is not None:
        extracted_lines = lines[start_index:]
    else:
        print("Start marker not found.")
        return

    # Overwrite the file with the extracted content
    with open(filepath, 'w') as file:
        file.writelines(extracted_lines)

# Usage
extract_and_overwrite(output_filepath)
