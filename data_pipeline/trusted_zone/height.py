#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 22:08:47 2025

@author: ceciliaperez
"""
import psycopg2
import csv

if __name__ == '__main__':

    
    hostname = "localhost"
    database = "terrassa_model"
    username = "ceciliaperez"
    pwd = ""
    port_id = 5432
    
    cur = None,
    conn = None,
    try: 
        conn = psycopg2.connect(  # Use 'conn' to conect .py to PostgreSQL
            host=hostname,
            dbname=database,
            user=username,
            password=pwd,
            port=port_id)
        cur = conn.cursor() 
        # datetype_query = """  
        #       ALTER TABLE catastro.tipo13_su ALTER COLUMN anyo_construccion TYPE date USING to_date(anyo_construccion::text, 'YYYY');
        # """
        # cur.execute(datetype_query)
        select = """
                SELECT c.gmlid, b.measured_height
                FROM citydb.cityobject c, citydb.building b
                WHERE b.id = c.id
                GROUP BY c.gmlid, b.id
        """
        cur.execute(select)
        table = cur.fetchall()
        
        csv_file_path = "/Users/ceciliaperez/Documents/UPC- MD/TFM/Data/Preprocess_data/height.csv"
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["gmlid", "measured_height"])  # Write header
            writer.writerows(table)  # Write data

        print(f"Data exported to {csv_file_path}")

    except Exception as error:
        print(f"Error: {error}")

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()