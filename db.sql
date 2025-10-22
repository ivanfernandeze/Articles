-- Esquema de base de datos Postgres para el asistente de investigación

CREATE TABLE IF NOT EXISTS articles (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  authors TEXT,
  year INT,
  url TEXT,
  doi TEXT,
  abstract TEXT,
  source TEXT,
  relevant BOOLEAN DEFAULT FALSE,
  summary TEXT,
  findings TEXT,
  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_articles_doi ON articles (doi);
CREATE INDEX IF NOT EXISTS idx_articles_year ON articles (year);
CREATE INDEX IF NOT EXISTS idx_articles_relevant ON articles (relevant);