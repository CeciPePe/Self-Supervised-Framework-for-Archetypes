# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:44:07 2023

@author: cperez
"""

# -*- coding: utf-8 -*-
import psycopg2

if __name__ == '__main__':
    # hostname = "172.16.27.100"
    # database = "ARV_district_buildings"
    # username = "postgres"
    # pwd = "GeoTerm2023@@"
    # port_id = 5432
    
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
            SELECT b.id, ST_ZMax(ST_3DExtent(s.geometry)) - ST_ZMin(ST_3DExtent(s.geometry)) AS height 
            FROM citydb.surface_geometry s, citydb.cityobject c, citydb.building b
            WHERE c.id = s.cityobject_id AND b.id = c.id
            GROUP BY b.id
        """
        cur.execute(select)
        table = cur.fetchall()
        for i in table:
            query_id = i[0]
            insert ="""
                UPDATE citydb.building
                SET measured_height = %s
                WHERE id = %s
            """
            cur.execute(insert, (i[1], i[0]))
        conn.commit()
        
    except Exception as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()