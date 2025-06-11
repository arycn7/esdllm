
import re

def remove_section(markdown_text: str, section_title: str) -> str:

    lines = markdown_text.splitlines()
    start_index = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells and cells[0] == section_title:
                start_index = i
                break

    if start_index is None:
        return markdown_text

    end_index = start_index + 1
    for j in range(start_index + 1, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("|"):
            if stripped.startswith("|---"):
                end_index = j + 1
                continue
            cells = [cell.strip() for cell in lines[j].split("|") if cell.strip()]
            if cells:
                if cells[0] != "":
                    end_index = j
                    break
        else:
            if lines[j].strip() == "":
                end_index = j
                break
        end_index = j + 1

    new_lines = lines[:start_index] + lines[end_index:]
    return "\n".join(new_lines)


def dynamic_chunk_splitter(section_text):

    section_length_words = len(section_text.split())
    chunking_threshold_words = 200

    if section_length_words <= chunking_threshold_words:
        return [section_text]
    else:
        section_length_chars = len(section_text)
        min_chunk_size_chars = 100
        max_chunk_size_chars = 2000  
        size_factor = 0.2

        chunk_size_chars = int(section_length_chars * size_factor)
        chunk_size_chars = max(min_chunk_size_chars, chunk_size_chars)
        chunk_size_chars = min(max_chunk_size_chars, chunk_size_chars)
        chunk_size_chars = min(chunk_size_chars, section_length_chars)

        section_complexity = min(1.0, section_length_chars / 4000)
        overlap_percentage = 0.15 + (section_complexity * 0.10)
        overlap_size_chars = int(chunk_size_chars * overlap_percentage)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=overlap_size_chars,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        return splitter.split_text(section_text)
def extract_section(markdown_text: str, section_title: str) -> str:
   
    lines = markdown_text.splitlines()
    extracted_sections = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|"):
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells and cells[0] == section_title:
                # Found a matching header row.
                start_index = i
                end_index = start_index + 1
                # Determine the block for this occurrence.
                for j in range(start_index + 1, len(lines)):
                    stripped = lines[j].lstrip()
                    if stripped.startswith("|"):
                        # Skip markdown divider rows.
                        if stripped.startswith("|---"):
                            end_index = j + 1
                            continue
                        new_cells = [cell.strip() for cell in lines[j].split("|") if cell.strip()]
                        if new_cells:
                            # New section header detected; stop at this point.
                            end_index = j
                            break
                    else:
                        if lines[j].strip() == "":
                            end_index = j
                            break
                    end_index = j + 1

                if end_index <= start_index + 1:
                    header_cells = [cell.strip() for cell in lines[start_index].split("|") if cell.strip()]
                    content = " ".join(header_cells[1:]) if len(header_cells) > 1 else ""
                else:
                    # Process the content rows: remove table formatting and join cell contents.
                    processed_lines = []
                    for k in range(start_index + 1, end_index):
                        curr_line = lines[k]
                        if curr_line.lstrip().startswith("|"):
                            row_cells = [cell.strip() for cell in curr_line.split("|") if cell.strip()]
                            processed_lines.append(" ".join(row_cells))
                        else:
                            processed_lines.append(curr_line.strip())
                    content = "\n".join(processed_lines)
                
                extracted_sections.append(content)
                i = end_index
                continue
        i += 1

    return "\n".join(extracted_sections).strip()
import re


def remove_markdown_elements(text: str) -> str:
    text = text.replace("<br>", " ")
    text = text.replace("•", "")
    text = " ".join(text.split())
    return text

def parse_and_extract(text):
  #cleanup (some)
  new_markdown = remove_section(text, "Module Pre-requisite")
  new_markdown = remove_section(new_markdown,"""Assessment Details2<br>Please include the following:<br>• Assessment<br>Component<br>• Assessment description<br>• Learning Outcome(s)<br>addressed<br>• % of total<br>• Assessment due date""")
  new_markdown = remove_section(new_markdown,"---")
  #extractions: (todo: process teaching and learning methods)
  modulecode=extract_section(new_markdown,"Module Code")
  modulename=extract_section(new_markdown,"Module Name")
  modulecontent=extract_section(new_markdown,"Module Content")
  moduleLOs=extract_section(new_markdown,"Module Learning Outcomes with reference<br>to the Graduate Attributes and how they<br>are developed in discipline")
  moduleassesment=extract_section(new_markdown,"""Assessment Details2<br>Please include the following:<br>• Assessment<br>Component<br>• Assessment description<br>• Learning Outcome(s)<br>addressed<br>• % of total<br>• Assessment due date""")
  moduleassesment=" FORMAT : Assessment Component,Assessment description, Learning Outcome(s) addressed, % of total, Assessment due date"+moduleassesment
  
  info=[modulecode,modulename,modulecontent,moduleLOs,moduleassesment]
  for i in range(len(info)):
    info[i]=remove_markdown_elements(info[i])
  modulecode=info[0]
  modulename=info[1]
  modulecontent=info[2]
  moduleLOs=info[3]
  moduleassesment=info[4]
  return modulecode,modulename,modulecontent,moduleLOs,moduleassesment
