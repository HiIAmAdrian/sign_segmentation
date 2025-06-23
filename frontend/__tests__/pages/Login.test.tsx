import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Login from '@/pages/Login';
import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import { Toaster } from '@/components/ui/toaster';

const mockLogin = vi.fn();
const mockNavigate = vi.fn();

vi.mock('@/contexts/AuthContext', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@/contexts/AuthContext')>();
    return {
        ...actual,
        useAuth: () => ({
            login: mockLogin,
            user: null,
            isAuthenticated: false,
            logout: vi.fn(),
        }),
    };
});

vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router-dom')>();
    return {
        ...actual,
        useNavigate: () => mockNavigate,
    };
});

const mockToastFnLogin = vi.fn();
vi.mock('@/components/ui/use-toast', () => ({
    useToast: () => ({
        toast: mockToastFnLogin,
    }),
}));


describe('Login Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const renderLogin = () => {
        return render(
            <BrowserRouter>
                <AuthProvider>
                    <Login />
                    <Toaster />
                </AuthProvider>
            </BrowserRouter>
        );
    };

    it('renders login form correctly', () => {
        renderLogin();
        expect(screen.getByLabelText(/Username/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/Password/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Login/i })).toBeInTheDocument();
    });

    it('allows typing in username and password fields', () => {
        renderLogin();
        const usernameInput = screen.getByLabelText(/Username/i);
        const passwordInput = screen.getByLabelText(/Password/i);

        fireEvent.change(usernameInput, { target: { value: 'testuser' } });
        fireEvent.change(passwordInput, { target: { value: 'password123' } });

        expect(usernameInput).toHaveValue('testuser');
        expect(passwordInput).toHaveValue('password123');
    });

    it('calls login function and navigates on successful login', async () => {
        mockLogin.mockResolvedValue(true);
        const { getByRole } = renderLogin();

        fireEvent.change(screen.getByLabelText(/Username/i), { target: { value: 'testuser' } });
        fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });

        await act(async () => {
            fireEvent.click(getByRole('button', { name: /Login/i }));
        });

        expect(mockLogin).toHaveBeenCalledWith('testuser', 'password123');
        await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith('/dashboard'));
    });

    it('shows error message on failed login', async () => {
        mockLogin.mockResolvedValue(false);
        const { getByRole } = renderLogin();

        fireEvent.change(screen.getByLabelText(/Username/i), { target: { value: 'testuser' } });
        fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });

        await act(async () => {
            fireEvent.click(getByRole('button', { name: /Login/i }));
        });

        expect(mockLogin).toHaveBeenCalledWith('testuser', 'password123');
        expect(mockNavigate).not.toHaveBeenCalled();

        expect(mockToastFnLogin).toHaveBeenCalledWith(expect.objectContaining({
            variant: 'destructive',
            title: 'Login failed'
        }));
    });

    it('disables login button while loading', async () => {
        mockLogin.mockImplementation(() => new Promise(resolve => setTimeout(() => resolve(true), 100)));
        const { getByRole } = renderLogin();

        fireEvent.change(screen.getByLabelText(/Username/i), { target: { value: 'testuser' } });
        fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });

        fireEvent.click(getByRole('button', { name: /Login/i }));

        expect(getByRole('button', { name: /Logging in.../i })).toBeDisabled();

        await act(async () => {
            await new Promise(resolve => setTimeout(resolve, 150));
        });
        expect(getByRole('button', { name: /Login/i })).not.toBeDisabled();
    });
});