// dbSetup.js
import 'dotenv/config';
import pg from 'pg';

const { PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE, PG_CONNECTION_STRING, PGSSL, PGSSLMODE } = process.env;

const useSSL =
  (PGSSL && PGSSL.toLowerCase() === 'require') ||
  (PGSSLMODE && PGSSLMODE.toLowerCase() === 'require');

export const pool = new pg.Pool(
  PG_CONNECTION_STRING
    ? {
        connectionString: PG_CONNECTION_STRING,
        ssl: useSSL ? { rejectUnauthorized: false } : undefined,
      }
    : {
        host: PGHOST,
        port: PGPORT ? parseInt(PGPORT, 10) : 5432,
        user: PGUSER,
        password: PGPASSWORD,
        database: PGDATABASE,
        ssl: useSSL ? { rejectUnauthorized: false } : undefined,
      }
);

console.log('DB connecting to', PGHOST, 'ssl:', useSSL);
