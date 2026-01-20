import { MongoClient, Db } from "mongodb";

const uri = process.env.MONGODB_URI!;
if (!uri) {
  throw new Error("❌ Missing MONGODB_URI");
}

let client: MongoClient;
let db: Db;

/**
 * Connect once and reuse
 */
export async function connectMongo(): Promise<Db> {
  if (db) return db;

  client = new MongoClient(uri);
  await client.connect();

  db = client.db("fintech-auth");

  console.log("✅ MongoDB connected");

  return db;
}
