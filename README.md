# Signature and Stamp Extraction Machine Learning Project

## Overview
This project is focused on the automated extraction of signatures and stamps from documents using machine learning techniques. The system intelligently identifies, isolates, and extracts these elements while handling overlapping cases between signatures and stamps.

## Project Workflow

1. **Dataset Creation and Annotation**
   - Created a custom dataset by annotating images
   - Each image was carefully labeled to identify signatures and stamps
   - Annotations served as the foundation for training our custom model

2. **Model Development and Training**
   - Designed and implemented a custom machine learning model
   - Trained the model on our annotated dataset
   - Fine-tuned model parameters for optimal performance
   - Generated `best.pt` weights file after successful training

3. **Feature Extraction**
   - Developed algorithms to crop and extract both signatures and stamps
   - Created separate extraction modules for each element type

4. **Overlap Resolution**
   - Implemented specialized techniques to address overlapping elements
   - Developed methods to separate signatures from stamps when they overlap
   - Successfully handled both cases: stamps overlapping signatures and signatures overlapping stamps

5. **Integration and Deployment**
   - Created `final.py` which integrates all components into a cohesive system
   - Developed a streamlined workflow from input document to extracted elements

## Getting Started

### Prerequisites
```
pip install -r requirements.txt
```

### Installation
```
git clone https://github.com/Rajar12345/Signature-and-Stamp-Extraction-Machine-Learning-Project.git
```

### Usage
```bash
python final.py --input document.jpg --output results/
```

## Results
The system successfully extracts signatures and stamps from documents, even in cases with overlapping elements, providing clean and separated outputs.

## Future Work
- Improving accuracy for complex documents
- Handling additional document elements like seals and handwritten text
- Adding support for batch processing
