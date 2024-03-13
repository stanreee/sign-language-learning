/* eslint-disable prettier/prettier */
import { MongoClient } from 'mongodb';

async function main() {
    // mongodb cluster access
    const uri = "mongodb+srv://CassieB:Azxsw21@aslingo.4cr3x99.mongodb.net/";
    /**
     * The Mongo Client you will use to interact with your database
     * See https://mongodb.github.io/node-mongodb-native/3.6/api/MongoClient.html for more details
     */
    const client = new MongoClient(uri);

    try {
        // Connect to the MongoDB cluster
        await client.connect();
        
        // Make the appropriate DB calls
        await createUser(client, "JJ99",
            {
                name: "James",
                username: "JJ99",
                password: "JamesJames",
                handedness: "Right"
            }
        );

        await findUserByInfo(client, "CassieB", "secondpass");

        await updatePassword(client, "CassieB", "secondpass", {password: "thisPassIsNew"});

        await deleteUserAccount(client, "JJ9")

    } catch (e) {
        console.error(e);
    } finally {
        // Close the connection to the MongoDB cluster
        await client.close();
    }
}

// Functions that make DB calls here
async function createUser(client, usernameOfUser, newUser){  // create new user from sign up info
    const doesUserExist = await client.db("ASLingo").collection("Users").findOne({username: usernameOfUser});

    if (doesUserExist) {  //if user exists, say they already have an account, and should sign in
        console.log(`User '${usernameOfUser}' already has an account`);
    } else {
        const result = await client.db("ASLingo").collection("Users").insertOne(newUser);
        console.log(`New user created with the following id: ${result.insertedId}`);
    }
}

async function findUserByInfo(client, usernameOfUser, passwordOfUser) {  // check if user exists
    const result = await client.db("ASLingo").collection("Users").findOne({ username: usernameOfUser, password: passwordOfUser });

    if (result) {
        console.log(`Found user '${usernameOfUser}':`);
        console.log(result);
    } else {
        console.log(`No user '${usernameOfUser}' found, please try again`);
    }
}

async function updatePassword(client, usernameOfUser, passwordOfUser, updatedPassword) {  // update existing user password
    const result = await client.db("ASLingo").collection("Users")
                        .updateOne({username: usernameOfUser, password: passwordOfUser }, { $set: updatedPassword });

    console.log(`${result.matchedCount} document(s) matched the query criteria.`);
    console.log(`${result.modifiedCount} document(s) was/were updated.`);
}

async function deleteUserAccount(client, usernameOfUser) {  // deletes a user account from database
    const result = await client.db("ASLingo").collection("Users")
            .deleteOne({ username: usernameOfUser });
    console.log(`${result.deletedCount} document(s) was/were deleted.`);
}

main().catch(console.error);