
import Dexie from 'dexie';

export interface User {
  id?: number;
  username: string;
  password: string;
}

class AppDatabase extends Dexie {
  users: Dexie.Table<User, number>;

  constructor() {
    super('SignSyncDB');
    this.version(1).stores({
      users: '++id, username, password',
    });
    this.users = this.table('users');
  }

  async initializeDefaultUser() {
    const userCount = await this.users.count();
    
    if (userCount === 0) {
      await this.users.add({
        username: 'demo',
        password: 'password', // In a real app, use hashed passwords
      });
    }
  }

  async authenticateUser(username: string, password: string): Promise<User | null> {
    const user = await this.users.where('username').equals(username).first();
    
    if (user && user.password === password) {
      return user;
    }
    
    return null;
  }
}

const db = new AppDatabase();
db.initializeDefaultUser();

export default db;
