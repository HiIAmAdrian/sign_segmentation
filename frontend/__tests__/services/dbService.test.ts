import { describe, it, expect, beforeEach, vi } from 'vitest';
import db, { User } from '@/services/dbService';
import { mockDexieTables } from '../setup';

const getMockedUsersTable = () => mockDexieTables['users'];

describe('dbService', () => {
    let usersTableMock: any;

    beforeEach(() => {
        vi.clearAllMocks();
        usersTableMock = getMockedUsersTable();

        usersTableMock.count.mockResolvedValue(0);
        usersTableMock.add.mockResolvedValue(1);
        usersTableMock.first.mockResolvedValue(undefined);
    });

    it('initializeDefaultUser should add a demo user if no users exist (relies on global mock setup)', async () => {
        usersTableMock.count.mockResolvedValue(0);
        await db.initializeDefaultUser();
        expect(usersTableMock.add).toHaveBeenCalledWith({ username: 'demo', password: 'password' });
    });

    it('initializeDefaultUser should not add a user if users already exist', async () => {
        usersTableMock.count.mockResolvedValue(1);

        await db.initializeDefaultUser();

        expect(usersTableMock.count).toHaveBeenCalledTimes(1);
        expect(usersTableMock.add).not.toHaveBeenCalled();
    });

    it('authenticateUser should return user on correct credentials', async () => {
        const mockUser: User = { id: 1, username: 'testuser', password: 'password123' };
        usersTableMock.first.mockResolvedValue(mockUser);

        const authenticatedUser = await db.authenticateUser('testuser', 'password123');

        expect(usersTableMock.where).toHaveBeenCalledWith('username');
        expect(usersTableMock.equals).toHaveBeenCalledWith('testuser');
        expect(usersTableMock.first).toHaveBeenCalledTimes(1);
        expect(authenticatedUser).toEqual(mockUser);
    });

    it('authenticateUser should return null if user not found', async () => {
        usersTableMock.first.mockResolvedValue(undefined);

        const authenticatedUser = await db.authenticateUser('nonexistent', 'password123');

        expect(authenticatedUser).toBeNull();
    });

    it('authenticateUser should return null on incorrect password', async () => {
        const mockUser: User = { id: 1, username: 'testuser', password: 'password123' };
        usersTableMock.first.mockResolvedValue(mockUser);

        const authenticatedUser = await db.authenticateUser('testuser', 'wrongpassword');

        expect(authenticatedUser).toBeNull();
    });
});