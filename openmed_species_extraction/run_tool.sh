#!/bin/bash
echo "OpenMed Species Extraction Tool"
echo "================================"
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Running species extraction..."
python openmed_species_extraction.py
echo ""
echo "Done! Check the output file: openmed_improved_fixed_Data_cleaned.csv"