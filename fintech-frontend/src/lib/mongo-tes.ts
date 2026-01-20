import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI!;
const client = new MongoClient(uri);

async function test() {
  await client.connect();
  await client.db("fintech-auth").command({ ping: 1 });
  console.log("✅ MongoDB Atlas connected successfully");
  await client.close();
}

test().catch((err) => {
  console.error("❌ MongoDB connection failed");
  console.error(err);
});
