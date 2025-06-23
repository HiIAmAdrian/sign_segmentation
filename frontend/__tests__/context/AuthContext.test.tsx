import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { AuthProvider, useAuth } from '../../src/contexts/AuthContext';
import db from '@/services/dbService';

vi.mock('@/services/dbService', () => ({
    default: {
        authenticateUser: vi.fn(),
    },
}));

const TestComponent = () => {
    const { user, isAuthenticated, login, logout } = useAuth();
    return (
        <div>
            <div data-testid="is-authenticated">{isAuthenticated.toString()}</div>
            <div data-testid="username">{user?.username || 'None'}</div>
            <button onClick={() => login('test', 'pass')}>Login</button>
            <button onClick={logout}>Logout</button>
        </div>
    );
};

describe('AuthContext', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('initial state should be unauthenticated', () => {
        render(
            <AuthProvider>
                <TestComponent />
            </AuthProvider>
        );
        expect(screen.getByTestId('is-authenticated')).toHaveTextContent('false');
        expect(screen.getByTestId('username')).toHaveTextContent('None');
    });

    it('login should authenticate user and update state on success', async () => {
        const mockUser = { id: 1, username: 'test', password: 'password' };
        (db.authenticateUser as ReturnType<typeof vi.fn>).mockResolvedValue(mockUser);

        render(
            <AuthProvider>
                <TestComponent />
            </AuthProvider>
        );

        await act(async () => {
            screen.getByText('Login').click();
        });

        expect(db.authenticateUser).toHaveBeenCalledWith('test', 'pass');
        expect(screen.getByTestId('is-authenticated')).toHaveTextContent('true');
        expect(screen.getByTestId('username')).toHaveTextContent('test');
    });

    it('login should not authenticate user on failure', async () => {
        (db.authenticateUser as ReturnType<typeof vi.fn>).mockResolvedValue(null);

        render(
            <AuthProvider>
                <TestComponent />
            </AuthProvider>
        );

        await act(async () => {
            screen.getByText('Login').click();
        });

        expect(screen.getByTestId('is-authenticated')).toHaveTextContent('false');
        expect(screen.getByTestId('username')).toHaveTextContent('None');
    });

    it('logout should deauthenticate user and clear state', async () => {
        const mockUser = { id: 1, username: 'test', password: 'password' };
        (db.authenticateUser as ReturnType<typeof vi.fn>).mockResolvedValue(mockUser);

        render(
            <AuthProvider>
                <TestComponent />
            </AuthProvider>
        );

        await act(async () => {
            screen.getByText('Login').click();
        });
        expect(screen.getByTestId('is-authenticated')).toHaveTextContent('true');

        await act(async () => {
            screen.getByText('Logout').click();
        });

        expect(screen.getByTestId('is-authenticated')).toHaveTextContent('false');
        expect(screen.getByTestId('username')).toHaveTextContent('None');
    });

    it('useAuth outside provider should throw error', () => {
        const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        expect(() => render(<TestComponent />)).toThrow('useAuth must be used within an AuthProvider');
        consoleErrorSpy.mockRestore();
    });
});