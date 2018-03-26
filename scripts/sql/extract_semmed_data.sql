-- Query used to extract initial data from SemMedDB.
-- USE semmedVER26_20160430
SELECT
    p.PMID,
    p.predicate,
    sp.INDICATOR_TYPE,
    sp.SUBJECT_TEXT,
    p.s_type,
    sp.SUBJECT_SCORE,
    sp.SUBJECT_DIST,
    sp.SUBJECT_MAXDIST,
    p.s_cui,
    sp.OBJECT_TEXT,
    p.o_type,
    sp.OBJECT_SCORE,
    sp.OBJECT_DIST,
    sp.OBJECT_MAXDIST,
    p.o_cui,
    s.SENTENCE,
    s.TYPE
FROM
    (SENTENCE_PREDICATION sp
	INNER JOIN PREDICATION_AGGREGATE p
	    ON sp.PREDICATION_ID=p.PID
        INNER JOIN SENTENCE s
            ON sp.SENTENCE_ID=s.SENTENCE_ID)
    INNER JOIN
        (CONCEPT c
            INNER JOIN CONCEPT_SEMTYPE cs 
    	        ON c.CONCEPT_ID=cs.CONCEPT_ID)
        ON c.CUI=p.s_cui OR c.CUI=p.o_cui
WHERE
    (p.s_type IN
        ('gngm',  -- Gene or Genome
         'aapp',  -- Amino Acid, Peptide, or Protein
	 'nnon',  -- Nucleic Acid, Nucleoside, or Nucleotide
         'phsu',  -- Pharmacologic Substance
         'clnd',  -- Clinical Drug
         'orch',  -- Organic Chemical
	 'vita',  -- Vitamin
	 'opco',  -- Organophosphorus Compound
	 'horm',  -- Hormone
	 'strd',  -- Steriod
	 'antb',  -- Antibiotic
	 'inch')  -- Inorganic Chemical
    ) 
    AND
    (p.o_type IN
        ('gngm',  -- Gene or Genome
         'aapp',  -- Amino Acid, Peptide, or Protein
	 'nnon',  -- Nucleic Acid, Nucleoside, or Nucleotide
         'phsu',  -- Pharmacologic Substance
         'clnd',  -- Clinical Drug
         'orch',  -- Organic Chemical
	 'vita',  -- Vitamin
	 'opco',  -- Organophosphorus Compound
	 'horm',  -- Hormone
	 'strd',  -- Steriod
	 'antb',  -- Antibiotic
	 'inch')  -- Inorganic Chemical
    ) 
    AND
    (p.predicate IN
        ('INHIBITS',
         'INTERACTS_WITH',
         'STIMULATES'))
LIMIT 1000000;
